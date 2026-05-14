#!/bin/bash

pretrained_model_dir=/mnt/afs/250010237/Emo-LiPO/pretrained_models/CosyVoice-300M-Instruct

# train llm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

# Stage 5: Train LLM
echo "Stage 5: Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
if [ "$train_engine" == 'deepspeed' ]; then
  echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
fi

cat data_ESD_intensity/train/parquet_sft/data.list > data_ESD_intensity/train_sft.data.list
cat data_ESD_intensity/dev/parquet_sft/data.list > data_ESD_intensity/dev_sft.data.list

for model in llm; do
  torchrun --nnodes=1 --nproc_per_node="$num_gpus" \
      --rdzv_id="$job_id" --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    cosyvoice/bin/train.py \
    --train_engine "$train_engine" \
    --config conf/cosyvoice_sft.yaml \
    --train_data data_ESD_intensity/train_sft.data.list \
    --cv_data data_ESD_intensity/dev_sft.data.list \
    --model "$model" \
    --checkpoint "$pretrained_model_dir/$model.pt" \
    --model_dir "$(pwd)/exp/cosyvoice_sft/$model/$train_engine" \
    --tensorboard_dir "$(pwd)/tensorboard/cosyvoice_sft/$model/$train_engine" \
    --ddp.dist_backend "$dist_backend" \
    --num_workers "${num_workers}" \
    --prefetch "${prefetch}" \
    --pin_memory \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer
done

echo "Stage 5 completed successfully!"
