export WANDB_API_KEY=190caefcc554590440e42593bfd6931f88f46f16
export WANDB_ENTITY=shuozhe
export HF_HUB_READ_TIMEOUT=60


# optimizer
optimizer="adamw"
weight_decay=0
grad_clipping=0.0
beta1=0.9
beta2=0.95
eps=0.000001
lr=0.0001
min_lr_ratio=0.1
scheduler="cosine"
warmup_steps=1000
num_training_steps=4_00_00
# max_train_tokens=1000000000
max_length=256

# training parameters

# batch_size: batch size per gpu
# total_batch_size = batch_size * nproc_per_node * num_nodes: total batch size for all GPUs
# gradient_accumulation = args.total_batch_size // (args.batch_size * world_size): number of steps to accumulate gradients

# 1 sec 1 step
CUDA_VISIBLE_DEVICES=0,1,2,3
WORLD_SIZE=4
batch_size=64
gradient_accumulation=2
total_batch_size=512

# model
model_config="/nfs/shuozhe/saved_model/Qwen2.5-0.5B"
continue_from=none
reinit_params=True
reinit_scope="all"  # options: all, embeddings, none
seed=42

# we only have 1 node, so nproc_per_node is the same as WORLD_SIZE or GPU count
nproc_per_node=$WORLD_SIZE

dtype="bfloat16"
activation_checkpointing=True

# dataset
# dataset="allenai/c4"
# dataset_config="en"
dataset="/nfs/shuozhe/saved_dataset/small-c4-dataset"
dataset_config="default"  # use None for no config


# log and eval
project_name="pretrain"
# project_name="test"
wandb_run_name="pythia-14m-SlimPajama-6B"
save_dir="./checkpoints/pythia-14m-SlimPajama-6B"
save_every=2000
eval_every=1000

export WORLD_SIZE=$WORLD_SIZE # total number of processes
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc_per_node $nproc_per_node pretrain.py \
        --model_config $model_config \
        --continue_from $continue_from \
        --batch_size $batch_size \
        --total_batch_size $total_batch_size \
        --max_length $max_length \
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
        --lr $lr \
        --scheduler $scheduler \
        --warmup_steps $warmup_steps \
        --num_training_steps $num_training_steps \
        --min_lr_ratio $min_lr_ratio \
        --reinit_params $reinit_params \
        --reinit_scope $reinit_scope \
        --seed $seed \
        --dataset $dataset \
        --dataset_config $dataset_config \
        --project_name $project_name \
        --wandb_run_name $wandb_run_name \