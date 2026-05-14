#!/bin/bash

pretrained_model_dir=../../../pretrained_models/CosyVoice-300M-Instruct

# Stage 1: Extract campplus speaker embedding
echo "Stage 1: Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data_ESD_intensity/$x dir"
for x in train dev test; do
  python tools/extract_embedding.py --dir data_ESD_intensity/$x \
    --onnx_path $pretrained_model_dir/campplus.onnx
done

echo "Stage 1 completed successfully!"