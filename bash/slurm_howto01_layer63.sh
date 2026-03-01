#!/bin/bash
#SBATCH --job-name=howto01_restricted_v2
#SBATCH --output=/home/tok/Emu3.5/outputs_howto/01_repot_houseplant_multilayer_restricted_v2/slurm_%j.log
#SBATCH --error=/home/tok/Emu3.5/outputs_howto/01_repot_houseplant_multilayer_restricted_v2/slurm_%j.log
#SBATCH -p nlp
#SBATCH -A nlp
#SBATCH --nodelist=nlp-h200-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00

mkdir -p /home/tok/Emu3.5/outputs_howto/01_repot_houseplant_multilayer_restricted_v2

export HF_HOME=/home/tok/.cache/huggingface

cd /home/tok/Emu3.5
conda run -n emu3p5 --no-capture-output python inference.py \
    --cfg configs/howto_01_layer63.py

conda run -n emu3p5 --no-capture-output python src/utils/vis_proto.py \
    --input /home/tok/Emu3.5/outputs_howto/01_repot_houseplant_multilayer_restricted_v2/proto/
