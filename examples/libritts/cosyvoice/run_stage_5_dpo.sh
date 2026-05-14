#!/bin/bash

# train llm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

# Stage 5: Train LLM
echo "Stage 5: Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
if [ $train_engine == 'deepspeed' ]; then
  echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
fi

cat data_ESD_intensity/train/parquet_dpo_2/data.list > data_ESD_intensity/train_dpo_2.data.list
cat data_ESD_intensity/dev/parquet_dpo_2/data.list > data_ESD_intensity/dev_dpo_2.data.list

for model in llm; do
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    cosyvoice/bin/train_dpo.py \
    --train_engine $train_engine \
    --config conf/cosyvoice_dpo.yaml \
    --train_data data_ESD_intensity/train_dpo_2.data.list \
    --cv_data data_ESD_intensity/dev_dpo_2.data.list \
    --model $model \
    --checkpoint exp/cosyvoice_sft/llm/torch_ddp/epoch_12_whole.pt \
    --model_dir `pwd`/exp/cosyvoice_dpo4/$model/$train_engine \
    --tensorboard_dir `pwd`/tensorboard/cosyvoice_dpo4/$model/$train_engine \
    --ddp.dist_backend $dist_backend \
    --num_workers ${num_workers} \
    --prefetch ${prefetch} \
    --pin_memory \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer
done

echo "Stage 5 completed successfully!"