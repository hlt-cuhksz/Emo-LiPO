#!/bin/bash

# Stage 3: Prepare required parquet format data
echo "Stage 3: Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
for x in train test dev; do
  mkdir -p data_ESD_intensity/$x/parquet_lipo_5_slightly
  python tools/make_parquet_list_lipo_5.py --num_utts_per_parquet 1000 \
    --num_processes 1 \
    --src_dir data_ESD_intensity/$x \
    --des_dir data_ESD_intensity/$x/parquet_lipo_5_slightly \
    --extra_intensity Slightly
done

echo "Stage 3 completed successfully!"