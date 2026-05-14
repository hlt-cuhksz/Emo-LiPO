#!/bin/bash
#SBATCH --job-name=stage_4_sft
#SBATCH --partition gpu3node
#SBATCH --gres=gpu:1
#SBATCH -o stage_4_sft.out
#SBATCH -t 2-00:00:00
#SBATCH --mem=32G

source ~/.bashrc
conda activate cosyvoice
export PYTHONPATH="$PWD/third_party/Matcha-TTS:$PYTHONPATH"

# Stage 4: Inference
echo "Stage 4: Run inference. Please make sure utt in tts_text is in prompt_data"
python cosyvoice/bin/inference_sft.py \
  --gpu 0 \
  --emotion_data utt2emo \
  --tts_text text.json \
  --model_dir pretrained_models/CosyVoice-300M-Instruct_sft \
  --result_dir result/Audio_sft

echo "Stage 4 completed successfully!"