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
num_training_steps=none
max_train_tokens=6_000_000_000
max_length=256

# training parameters

# batch_size: batch size per gpu
# total_batch_size = batch_size * nproc_per_node * num_nodes: total batch size for all GPUs
# gradient_accumulation = args.total_batch_size // (args.batch_size * world_size): number of steps to accumulate gradients

# 1 sec 1 step
CUDA_VISIBLE_DEVICES=0,1,2,3
WORLD_SIZE=4
batch_size=256
gradient_accumulation=2
total_batch_size=2048
model_config="EleutherAI/pythia-14m"
continue_from=none
reinit_params=True
# model_config="/nfs/shuozhe/clean_pretrain/checkpoints/test_load/pythia-14m-SlimPajama-6B/model_20"
# continue_from="/nfs/shuozhe/clean_pretrain/checkpoints/test_load/pythia-14m-SlimPajama-6B/model_20"
# reinit_params=False
reinit_scope="all"  # options: all, embeddings, none
seed=42

# we only have 1 node, so nproc_per_node is the same as WORLD_SIZE or GPU count
nproc_per_node=$WORLD_SIZE

dtype="bfloat16"
activation_checkpointing=True

# dataset
dataset="DKYoon/SlimPajama-6B"
dataset_config="default"

# log and eval
project_name="pretrain"
# project_name="test"
wandb_run_name="pythia-14m-SlimPajama-6B"
save_dir="./checkpoints/pythia-14m-SlimPajama-6B"
save_every=2000
eval_every=1000
# save_every=10
# eval_every=10

export WORLD_SIZE=$WORLD_SIZE # total number of processes
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc_per_node $nproc_per_node pretrain.py \
        --model_config $model_config \
        --continue_from $continue_from \
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
        --lr $lr \
        --scheduler $scheduler \
        --warmup_steps $warmup_steps \
        --num_training_steps $num_training_steps \
        --max_train_tokens $max_train_tokens \
        --min_lr_ratio $min_lr_ratio \
        --reinit_params $reinit_params \
        --reinit_scope $reinit_scope \
        --seed $seed \
        --dataset $dataset \
        --dataset_config $dataset_config \
        --project_name $project_name \
        --wandb_run_name $wandb_run_name \