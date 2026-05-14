#!/bin/bash
#SBATCH --partition gpu3node
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mem=32G

# 获取命令行输入的 llm_value 参数
LLM_VALUE=$1

# 检查 llm_value 是否提供
if [ -z "$LLM_VALUE" ]; then
  echo "Error: llm_value is required as the first argument."
  exit 1
fi

source ~/.bashrc
conda activate cosyvoice
export PYTHONPATH="$PWD/third_party/Matcha-TTS:$PYTHONPATH"

# Stage 4: Inference
echo "Stage 4: Run inference with llm_value: $LLM_VALUE. Please make sure utt in tts_text is in prompt_data"
python cosyvoice/bin/inference_lipo.py \
  --gpu 0 \
  --emotion_data utt2emo \
  --tts_text text.json \
  --model_dir pretrained_models/CosyVoice-300M-Instruct_lipo \
  --llm $LLM_VALUE \
  --result_dir result/Audio_lipo_$LLM_VALUE     # 动态修改 result 目录

echo "Stage 4 completed successfully!"
