export WANDB_API_KEY=190caefcc554590440e42593bfd6931f88f46f16
export WANDB_ENTITY=shuozhe


# optimizer
optimizer="adam"
weight_decay=0
grad_clipping=1.0
beta1=0.9
beta2=0.95
eps=0.00001

# training parameters

# batch_size: batch size per gpu
# total_batch_size = batch_size * nproc_per_node * num_nodes: total batch size for all GPUs
# gradient_accumulation = args.total_batch_size // (args.batch_size * world_size): number of steps to accumulate gradients
# memory
# batch_size=2
# total_batch_size=8
# gradient_accumulation=1
# WORLD_SIZE=4
# nproc_per_node=4

CUDA_VISIBLE_DEVICES=2,3
WORLD_SIZE=1
batch_size=2
total_batch_size=2
gradient_accumulation=1

# we only have 1 node, so nproc_per_node is the same as WORLD_SIZE or GPU count
nproc_per_node=$WORLD_SIZE

dtype="bfloat16"
activation_checkpointing=True

# model
model_config="/nfs/shuozhe/saved_model/Qwen2.5-0.5B"
dataset="allenai/c4"

# log and eval
project_name="pretrain_test"
wandb_run_name="test_run"
save_dir="test"
save_every=128
eval_every=128

export WORLD_SIZE=$WORLD_SIZE # total number of processes
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc_per_node $nproc_per_node pretrain.py \
        --model_config $model_config \
        --batch_size $batch_size \
        --total_batch_size $total_batch_size \
        --activation_checkpointing $activation_checkpointing \
        --gradient_accumulation $gradient_accumulation \
        --save_dir $save_dir \
        --save_every $save_every \
        --eval_every $eval_every \
        --dtype $dtype \
        --optimizer $optimizer \
        --weight_decay $weight_decay \
        --grad_clipping $grad_clipping \
        --beta1 $beta1 \
        --beta2 $beta2 \
        --eps $eps \