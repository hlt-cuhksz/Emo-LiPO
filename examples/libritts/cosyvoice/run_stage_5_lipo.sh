#!/bin/bash

# 通用训练脚本
# 用法: ./train_lipo.sh <type> <length> <lambda_mode>
# 示例: ./train_lipo.sh rule 7 index rule
#       ./train_lipo.sh pitch 7 score pitch
#       ./train_lipo.sh energy 7 no_lambda energy

# 参数检查
if [ $# -ne 4 ]; then
    echo "用法: $0 <type> <length> <lambda_mode> <data>"
    echo "  type:        rule | pitch | energy"
    echo "  length:      4 | 5 | 6 | 7"
    echo "  lambda_mode: index | score | no_lambda"
    echo ""
    echo "示例: $0 rule 7 index"
    exit 1
fi

TYPE=$1
LENGTH=$2
LAMBDA_MODE=$3
DATA=$4

# 验证参数
if [[ ! "$TYPE" =~ ^(rule|pitch|energy|rule_with_neutral)$ ]]; then
    echo "错误: type 必须是 rule, pitch, energy 或 rule_with_neutral"
    exit 1
fi

if [[ ! "$LENGTH" =~ ^(4|5|6|7)$ ]]; then
    echo "错误: length 必须是 4, 5, 6 或 7"
    exit 1
fi

if [[ ! "$LAMBDA_MODE" =~ ^(index|score|no_lambda)$ ]]; then
    echo "错误: lambda_mode 必须是 index, score 或 no_lambda"
    exit 1
fi

# 基础配置
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

# 构建实验名称
EXP_NAME="cosyvoice_lipo_${TYPE}_${LENGTH}_${LAMBDA_MODE}"

echo "============================================"
echo "训练配置:"
echo "  Type:        $TYPE"
echo "  Length:      $LENGTH"
echo "  Lambda Mode: $LAMBDA_MODE"
echo "  Exp Name:    $EXP_NAME"
echo "  GPUs:        $num_gpus"
echo "============================================"

# Stage 5: Train LLM
echo "Stage 5: Run train. We only support llm training for now. If you want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
fi

# 准备数据列表
cat data_ESD_intensity/train/parquet_lipo_${DATA}/data.list > data_ESD_intensity/train_lipo_${DATA}.data.list
cat data_ESD_intensity/dev/parquet_lipo_${DATA}/data.list > data_ESD_intensity/dev_lipo_${DATA}.data.list

# 构建 lambda 相关参数
LAMBDA_ARGS=""
if [ "$LAMBDA_MODE" != "no_lambda" ]; then
    LAMBDA_ARGS="--use_lambda --lambda_type ${LAMBDA_MODE}"
fi

# 运行训练
for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
        cosyvoice/bin/train_lipo.py \
        --train_engine $train_engine \
        --config conf/cosyvoice_lipo.yaml \
        --train_data data_ESD_intensity/train_lipo_${DATA}.data.list \
        --cv_data data_ESD_intensity/dev_lipo_${DATA}.data.list \
        --model $model \
        --checkpoint exp/cosyvoice_sft/llm/torch_ddp/epoch_12_whole.pt \
        --model_dir `pwd`/exp/${EXP_NAME}_${DATA}_5/$model/$train_engine \
        --tensorboard_dir `pwd`/tensorboard/${EXP_NAME}_${DATA}_5/$model/$train_engine \
        --ddp.dist_backend $dist_backend \
        --num_workers ${num_workers} \
        --prefetch ${prefetch} \
        --pin_memory \
        --deepspeed_config ./conf/ds_stage2.json \
        --deepspeed.save_states model+optimizer \
        --length ${LENGTH} \
        ${LAMBDA_ARGS}
done

echo "Stage 5 completed successfully!"
echo "实验 ${EXP_NAME} 训练完成!"
