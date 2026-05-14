#!/bin/bash

pretrained_model_dir=../../../pretrained_models/CosyVoice-300M-Instruct

# Stage 2: Extract discrete speech token
echo "Stage 2: Extract discrete speech token, you will get utt2speech_token.pt in data_ESD_intensity/$x dir"
for x in train test dev; do
  python tools/extract_speech_token.py --dir data_ESD_intensity/$x \
    --onnx_path $pretrained_model_dir/speech_tokenizer_v1.onnx
done

echo "Stage 2 completed successfully!"