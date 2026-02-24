# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib as imp
import os
import os.path as osp
from pathlib import Path
import random
import time
from time import sleep
from PIL import Image
import torch
from tqdm import tqdm

from src.utils.model_utils import build_emu3p5
from src.utils.generation_utils import generate, multimodal_decode, decode_image
from src.utils.painting_utils import ProtoWriter
from src.utils.input_utils import build_image, smart_resize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="", type=str)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--worker_id", default=0, type=int)
    args = parser.parse_args()
    return args

def inference(
    cfg,
    model,
    tokenizer,
    vq_model,
):
    save_path = cfg.save_path

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/proto", exist_ok=True)
    proto_writer = ProtoWriter()

    # device_map="auto" can put model.device on "meta"; find the first real device instead
    actual_device = next(
        (p.device for p in model.parameters() if p.device.type != "meta"),
        torch.device("cuda"),
    )

    for name, question in tqdm(cfg.prompts, total=len(cfg.prompts)):
        if osp.exists(f"{save_path}/proto/{name}.pb"):
            print(f"[WARNING] Result already exists, skipping {name}", flush=True)
            continue

        torch.cuda.empty_cache()

        reference_image = None
        if not isinstance(question, str):
            if isinstance(question["reference_image"], list):
                print(f"[INFO] {len(question['reference_image'])} reference images are provided")
                reference_image = []
                for img in question["reference_image"]:
                    reference_image.append(Image.open(img).convert("RGB"))
            else:
                print (f"[INFO] 1 reference image is provided")
                reference_image = Image.open(question["reference_image"]).convert("RGB")
            question = question["prompt"]
        else:
            print(f"[INFO] No reference image is provided")
        
        proto_writer.clear()
        proto_writer.extend([["question", question]])
        if reference_image is not None:
            if isinstance(reference_image, list):
                for idx, img in enumerate(reference_image):
                    proto_writer.extend([[f"reference_image", img]])
            else:
                proto_writer.extend([["reference_image", reference_image]])

        success = True
        prompt = cfg.template.format(question=question)

        print(f"[INFO] Handling prompt: {prompt}")
        if reference_image is not None:
            if isinstance(reference_image, list):
                image_str = ""
                for img in reference_image:
                    image_str += build_image(img, cfg, tokenizer, vq_model)
            else:
                image_str = build_image(reference_image, cfg, tokenizer, vq_model)
            prompt = prompt.replace("<|IMAGE|>", image_str)
            unc_prompt = cfg.unc_prompt.replace("<|IMAGE|>", image_str)
        else:
            unc_prompt = cfg.unc_prompt

        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(actual_device)

        if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
            BOS = torch.Tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([BOS, input_ids], dim=1)

        unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(actual_device)

        if hasattr(cfg, "img_unc_prompt"):
            full_unc_ids = tokenizer.encode(cfg.img_unc_prompt, return_tensors="pt", add_special_tokens=False).to(actual_device)
        else:
            full_unc_ids = None

        force_same_image_size = True
        # for x2i task, if multiple reference images are provided as a list, force_same_image_size should be False
        if isinstance(reference_image, list) and len(reference_image) > 1:
            force_same_image_size = False   
        
        t_gen_start = time.perf_counter()
        # Force streaming so each image is decoded and written as soon as it is generated,
        # rather than waiting for the entire sequence to complete.
        orig_streaming = getattr(cfg, "streaming", False)
        cfg.streaming = True
        text_buf = ""
        img_idx = 0
        _img_buf_offset = 0  # tracks remap_image_str consumption across images
        _layer_tag = getattr(cfg, "image_logits_layer", None)
        preview_dir = osp.join(save_path, "preview")
        os.makedirs(preview_dir, exist_ok=True)
        try:
            for ev in generate(cfg, model, tokenizer, input_ids, unconditional_ids, full_unc_ids, force_same_image_size):
                if ev["type"] == "text":
                    text_buf += ev["text"]
                elif ev["type"] == "image":
                    if text_buf:
                        proto_writer.extend(multimodal_decode(text_buf, tokenizer, vq_model))
                        text_buf = ""
                    raw_img_str = ev["image"]
                    pending = img_idx + 1
                    tok_count = raw_img_str.count("<|") if isinstance(raw_img_str, str) else "?"
                    print(f"[INFO] Image #{pending}: received {tok_count} tokens", flush=True)

                    # --- Image A: final-layer decode (vanilla generation result) ---
                    print(f"[INFO] Image #{pending} decoding final-layer...", flush=True)
                    img_final = decode_image(raw_img_str, tokenizer, vq_model) if isinstance(raw_img_str, str) else raw_img_str
                    if img_final is not None:
                        proto_writer.extend([("image", img_final)])
                        img_idx += 1
                        img_final.save(osp.join(preview_dir, f"{name}_{img_idx:02d}_final.png"))
                        print(f"[INFO] Image #{img_idx} (final) saved → preview/{name}_{img_idx:02d}_final.png  ({img_final.width}x{img_final.height}px)", flush=True)
                    elif proto_writer.story.clips:
                        print(f"[WARNING] Image #{pending} final-layer decode failed — skipping", flush=True)
                        proto_writer._get_last_image()

                    # --- Image B: intermediate-layer decode (logit-lens remap) ---
                    if isinstance(raw_img_str, str) and _layer_tag is not None and hasattr(model, "remap_image_str") and model._image_layer_hs_buf:
                        remapped_str, _img_buf_offset = model.remap_image_str(raw_img_str, tokenizer, _img_buf_offset)
                        print(f"[INFO] Image #{pending} decoding layer-{_layer_tag}...", flush=True)
                        img_inter = decode_image(remapped_str, tokenizer, vq_model)
                        if img_inter is not None:
                            img_inter.save(osp.join(preview_dir, f"{name}_{img_idx:02d}_layer{_layer_tag}.png"))
                            print(f"[INFO] Image #{img_idx} (layer {_layer_tag}) saved → preview/{name}_{img_idx:02d}_layer{_layer_tag}.png", flush=True)
                        else:
                            print(f"[WARNING] Image #{pending} layer-{_layer_tag} decode failed", flush=True)

                elif ev["type"] == "broken_image":
                    pass  # normal intermediate streaming progress event, not an error
                # "final_ids" and other internal events are ignored
            if text_buf:
                proto_writer.extend(multimodal_decode(text_buf, tokenizer, vq_model))
        except Exception as e:
            success = False
            print(f"[ERROR] Failed during generation: {e}")
        finally:
            cfg.streaming = orig_streaming
        t_gen_end = time.perf_counter()
        generate_secs = t_gen_end - t_gen_start
        print(f"[TIMING] generate: {generate_secs:.1f}s ({generate_secs/60:.1f}min)", flush=True)

        if not success:
            continue

        proto_writer.save(f"{save_path}/proto/{name}.pb")

def main():
    args = parse_args()
    cfg_name = Path(args.cfg).stem
    cfg_package = Path(args.cfg).parent.__str__().replace("/", ".")
    cfg = imp.import_module(f".{cfg_name}", package=cfg_package)

    rank, world_size = args.worker_id, args.num_workers

    cfg.rank = rank
    cfg.world_size = world_size

    if isinstance(cfg.prompts, dict):
        cfg.prompts = [(n, p) for n, p in cfg.prompts.items()]
    else:
        cfg.prompts = [(f"{idx:03d}", p) for idx, p in enumerate(cfg.prompts)]

    cfg.prompts = [(n, p) for n, p in cfg.prompts if not osp.exists(f"{cfg.save_path}/proto/{n}.pb")]
    cfg.prompts = cfg.prompts[rank::world_size]
    cfg.num_prompts = len(cfg.prompts)

    hf_device, vq_device = cfg.hf_device, cfg.vq_device

    t_load_start = time.perf_counter()
    model, tokenizer, vq_model = build_emu3p5(
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        model_device=hf_device,
        vq_device=vq_device,
        image_logits_layer=getattr(cfg, "image_logits_layer", None),
        **getattr(cfg, "diffusion_decoder_kwargs", {}),
    )
    t_load_end = time.perf_counter()
    load_secs = t_load_end - t_load_start
    print(f"[TIMING] model load: {load_secs:.1f}s ({load_secs/60:.1f}min)", flush=True)
    print(f"[INFO] Model loaded successfully")
    cfg.special_token_ids = {}
    for k, v in cfg.special_tokens.items():
        cfg.special_token_ids[k] = tokenizer.encode(v)[0]

    random.seed(cfg.seed + rank)

    t_infer_start = time.perf_counter()
    inference(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        vq_model=vq_model,
    )
    t_infer_end = time.perf_counter()
    infer_secs = t_infer_end - t_infer_start
    total_secs = t_load_end - t_load_start + infer_secs
    print(f"[TIMING] inference (all prompts): {infer_secs:.1f}s ({infer_secs/60:.1f}min)", flush=True)
    print(f"[TIMING] total: {total_secs:.1f}s ({total_secs/60:.1f}min)", flush=True)
    print(f"[INFO] Inference finished")

    # Write timing metadata
    os.makedirs(cfg.save_path, exist_ok=True)
    meta_path = osp.join(cfg.save_path, "timing_metadata.txt")
    with open(meta_path, "w") as f:
        f.write(f"model_load_sec: {load_secs:.1f}\n")
        f.write(f"model_load_min: {load_secs/60:.2f}\n")
        f.write(f"inference_all_prompts_sec: {infer_secs:.1f}\n")
        f.write(f"inference_all_prompts_min: {infer_secs/60:.2f}\n")
        f.write(f"total_sec: {total_secs:.1f}\n")
        f.write(f"total_min: {total_secs/60:.2f}\n")
        f.write(f"num_prompts: {cfg.num_prompts}\n")
    print(f"[INFO] Timing saved to {meta_path}", flush=True)

if __name__ == "__main__":
    main()
