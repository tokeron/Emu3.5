# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from src.utils.logging_utils import setup_logger
cfg_name = Path(__file__).stem

model_path = "BAAI/Emu3.5"
vq_path = "BAAI/Emu3.5-VisionTokenizer"
tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

task_type = "howto"
use_image = False

# Depth-freeze intervention: during image generation, hidden states are frozen
# at this layer — all later layers output the same representation.
# Text generation uses full depth (all 64 layers).
image_stop_layer = 40

exp_name = "01_repot_houseplant_depth_freeze_L40"
save_path = f"./outputs_howto/{exp_name}"
save_to_proto = True
setup_logger(save_path)

hf_device = "auto"
vq_device = "cuda:0"
streaming = False
unconditional_type = "no_text"
classifier_free_guidance = 3.0
max_new_tokens = 40000
image_area = 518400

def build_unc_and_template(task: str, with_image: bool):
    task_str = task.lower()
    if task_str in {"howto", "story", "explore", "vla"}:
        extra_system_prompt = " Please generate a response with interleaved text and images."
    else:
        extra_system_prompt = ""
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task.%s USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>" % (task_str, extra_system_prompt)
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task.%s USER: {question} ASSISTANT: <|extra_100|>" % (task_str, extra_system_prompt)
    return unc_p, tmpl

unc_prompt, template = build_unc_and_template(task_type, use_image)

sampling_params = dict(
    use_cache=True,
    text_top_k=200,
    text_top_p=0.8,
    text_temperature=0.7,
    image_top_k=10240,
    image_top_p=1.0,
    image_temperature=1.0,
    top_k=131072,
    top_p=1.0,
    temperature=1.0,
    num_beams_per_group=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    max_new_tokens=max_new_tokens,
    guidance_scale=1.0,
    use_differential_sampling=True,
)

sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1
sampling_params["num_beams"] = sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]

special_tokens = dict(
    BOS="<|extra_203|>", EOS="<|extra_204|>", PAD="<|endoftext|>",
    EOL="<|extra_200|>", EOF="<|extra_201|>", TMS="<|extra_202|>",
    IMG="<|image token|>", BOI="<|image start|>", EOI="<|image end|>",
    BSS="<|extra_100|>", ESS="<|extra_101|>", BOG="<|extra_60|>",
    EOG="<|extra_61|>", BOC="<|extra_50|>", EOC="<|extra_51|>",
)

seed = 6666

prompts = [
    'Create a step-by-step visual guide on how to repot a small houseplant. For each major stage of the process, generate a clear instruction in text followed by a corresponding image that demonstrates that specific action. The guide should cover: preparing the new pot, removing the plant from its old pot, loosening the roots, placing it in the new soil, and the final watering.'
]
