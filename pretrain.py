import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
from datasets import load_dataset

import datasets
import datasets.distributed
from dataclasses import dataclass
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
# from peft_pretraining.modeling_llama import LlamaForCausalLM

import bitsandbytes as bnb
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor

transformers.logging.set_verbosity_error()

@dataclass
class ModelConfig:
    num_layers: int  # number of transformer layers (blocks)
    n_head: int      # number of attention heads 
    hidden_dim: int  # hidden dimension
    vocab_size: int  # vocabulary size
    num_key_value_heads: int = None 
    max_seq_len: int = None   # max sequence length
    ffn_embed_dim: int = None # hidden dimension of FFN, default to 4 * hidden_dim
    model_type: str = None    # model type as tagged on Hugging Face (e.g., gpt2, opt, llama.)
    model_name: str = None    # model name as tagged on Hugging Face (e.g., gpt2-xl, opt, llama-13b.)
    
    def __post_init__(self):
        if self.num_key_value_heads is None: # 如果不存在，设置默认值
            self.num_key_value_heads = self.n_head 
            
        if self.ffn_embed_dim is None:
            self.ffn_embed_dim = self.hidden_dim * 4

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, default="meta-llama/Llama-2-7b", required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", type=bool, default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    # W&B related args
    parser.add_argument("--project_name", type=str, default="llm-pretraining")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--job_type", type=str, default="train")
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None, help="Set to resume a run by id")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    # beta1 for adafactor and beta2 for adamW
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-5)

    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")

    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")

    # training data parameters
    parser.add_argument("--dataset", type=str, default="allenai/c4")


    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(args, model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    was_training = model.training
    model.eval()
    try:
        _time = time.time()
        val_data = datasets.load_dataset(args.dataset, "en", split="validation", streaming=True) #DGX

        val_data = val_data.shuffle(seed=42)
        logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

        if not args.single_gpu:
            val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

        val_data_mapped = val_data.map(
            preprocess_batched,
            batched=True,
            remove_columns=["text", "timestamp", "url"],
        )
        val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)
        target_eval_tokens = 4096
        evaluated_on_tokens = 0
        total_loss = torch.tensor(0.0).to(device)
        # start from 0 so averaging isn't biased; guard division by zero below
        total_batches = 0
        logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

        for batch in val_data_mapped.batch(batch_size=batch_size):
            if evaluated_on_tokens > target_eval_tokens:
                break
            total_batches += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            loss = model(**batch, labels=labels).loss
            total_loss += loss.detach()

            evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

        total_loss = total_loss / max(1, total_batches)

        # Gather losses across all GPUs when distributed
        gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
        if (not args.single_gpu) and dist.is_available() and dist.is_initialized() and world_size > 1:
            dist.all_gather(gathered_losses, total_loss)
        else:
            gathered_losses = [total_loss]
        total_loss = sum([t.item() for t in gathered_losses]) / max(1, world_size)

        return total_loss, evaluated_on_tokens
    finally:
        if was_training:
            model.train()

def get_model_and_gpu_config_by_name(model_name="llama-13b", gpu_name="v100-pcie-32gb") -> dict:
    """Read model and gpu configs from a json file."""
    # config_files = ["configs/model_configs.json", "configs/gpu_configs.json"]
    config_files = ["configs/model_configs.json"]
    model_config, gpu_config = {}, {}
    
    for config_filename in config_files:
        with open(config_filename, "r") as f:
            config_json = json.load(f)
            
            if "model" in config_filename:
                assert model_name in config_json, f"model name {model_name} not found in {config_filename}"
                config_dict = config_json[model_name]
                model_config = ModelConfig(**config_dict)
            
            # elif "gpu" in config_filename:
            #     assert gpu_name in config_json, f"gpu name {gpu_name} not found in {config_filename}"
            #     config_dict = config_json[gpu_name]
            #     gpu_config = GPUConfig(**config_dict)
            else:
                assert False, f"unknown config type when reading: {type}"
            
    # return model_config, gpu_config
    return model_config

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def dist_train_assert_check(args):

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    # Note: GA/TBS validation is handled in main() after runtime init to support both single and distributed modes.

    # turn off logger
    if global_rank != 0: logger.remove()

    # NOTE: do not init W&B here. We initialize once in main after run_config is built.

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    return global_rank, local_rank, world_size, device

def init_runtime(args):
    """Initialize runtime for single-GPU or distributed modes.
    Returns (global_rank, local_rank, world_size, device).
    """
    if args.single_gpu:
        device = "cuda:0"
        torch.cuda.set_device(0)
        global_rank, local_rank, world_size = 0, 0, 1
        # NOTE: do not init W&B here. Initialization happens once in main with full config.

        logger.info("Single-GPU mode")
        logger.info("*" * 40)
        logger.info(f"Starting training with the arguments")
        for k, v in vars(args).items():
            logger.info(f"{k:30} {v}")
        logger.info("*" * 40)
        return global_rank, local_rank, world_size, device
    else:
        return dist_train_assert_check(args)

def tokenizer_data_loader(args, global_rank, world_size):

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained(args.model_config, model_max_length=args.max_length)

    # tokenizer.pad_token = "[PAD]"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    tokenizer.padding_side = "right"

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    data = datasets.load_dataset(args.dataset, "en", split="train", streaming=True, trust_remote_code=True)
    logger.info(f"Shuffling data with seed {args.seed}")
    data: datasets.Dataset = data.shuffle(seed=args.seed)

    # split the dataset by node
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )
    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    # For IterableDataset / streaming, persistent_workers can cause workers to hang or keep stale streams.
    # Disable persistent_workers for streaming workloads and let the user control num_workers via --workers.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.workers,
        persistent_workers=False,
    )

    return tokenizer, preprocess_batched, dataloader

def load_model(args, device):

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        torch_dtype = torch.bfloat16 if args.dtype in ["bf16", "bfloat16"] else None
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            args.continue_from, torch_dtype=torch_dtype
        )
        model_config = AutoConfig.from_pretrained(args.continue_from)

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)
    else:
        model_config = AutoConfig.from_pretrained(args.model_config)
        if args.use_hf_model:
            model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
        else:
            model = AutoModelForCausalLM.from_config(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    return model, model_config, global_step, update_step, beginning_step, tokens_seen, tokens_seen_before

def get_param_progress_bar(args, model, model_config, device, world_size, global_rank, update_step):
    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": args.dataset,
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    pbar = None
    # Do not update wandb.config or save here. W&B is initialized centrally with the full config.
    if global_rank == 0:
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    return trainable_params, run_config, pbar


def init_wandb_once(args, run_config, global_rank):
    """Initialize wandb one time on rank 0 and pass the full run_config.
    Returns the wandb Run object or None for non-rank0 processes."""
    if global_rank != 0:
        return None

    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None

    run = wandb.init(
        project=args.project_name,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.wandb_run_name or args.name,
        group=args.group,
        job_type=args.job_type,
        notes=args.notes,
        tags=tags,
        config=run_config,
        id=args.wandb_id,
        resume="allow",
        settings=wandb.Settings(code_dir="."),
    )

    # Make update_step the x-axis for train/* and eval/* metrics
    try:
        wandb.define_metric("update_step")
        wandb.define_metric("train/*", step_metric="update_step")
        wandb.define_metric("eval/*", step_metric="update_step")
    except Exception:
        # Older wandb versions may not have define_metric; ignore if it fails
        pass

    return run

def load_optimizer(args, model, trainable_params):

    galore_params = []
    optimizer_dict = {}
    if 'galore' in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        param_groups = [{'params': regular_params},
                        {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type}]

    layer_wise_flag = False
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adamw":
        optimizer = transformers.optimization.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)
    elif args.optimizer.lower() == "galore_adamw":
        # redefine way to call galore_adamw
        optimizer = GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)
    # implement sgd
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)
    # implement adafactor
    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # low-rank adafactor
    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # 8-bit Adam
    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)
    elif args.optimizer.lower() == "galore_adamw8bit":
        optimizer = GaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)
    elif args.optimizer.lower() == 'galore_adamw8bit_per_layer':
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit([{'params': [p], 'rank': args.rank, 'update_proj_gap': args.update_proj_gap * 2, 'scale': args.galore_scale, 'proj_type': args.proj_type}], lr=args.lr, weight_decay=args.weight_decay)
                else:
                    optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None:
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        layer_wise_flag = True

    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    scheduler = None
    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    return optimizer, layer_wise_flag, optimizer_dict, galore_params, scheduler
def check_train_assertions(args, world_size):
    # Validate GA/TBS after runtime init so world_size is known
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % (args.batch_size * world_size) == 0, \
                "total_batch_size must be divisible by (batch_size * world_size)"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"
        assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size
    else:
        assert args.gradient_accumulation is not None, \
            "When --total_batch_size is not provided, you must set --gradient_accumulation."

def main(args):
    set_seed(args.seed)
    global_rank, local_rank, world_size, device = init_runtime(args)
    # How often to push logs to wandb (1 = every update). Increasing this reduces stdout noise.
    LOG_EVERY = 1
    check_train_assertions(args, world_size)
    tokenizer, preprocess_batched, dataloader = tokenizer_data_loader(args, global_rank, world_size)
    model, model_config, global_step, update_step, beginning_step, tokens_seen, tokens_seen_before = load_model(args, device)
    # Ensure model embeddings match tokenizer vocabulary if we added a pad token
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        logger.warning(f"resize_token_embeddings call failed or unnecessary: {e}")

    trainable_params, run_config, pbar = get_param_progress_bar(args, model, model_config, device, world_size, global_rank, update_step)

    # Initialize W&B once (rank 0). We pass the full run_config here.
    wandb_run = init_wandb_once(args, run_config, global_rank)

    optimizer, layer_wise_flag, optimizer_dict, galore_params, scheduler = load_optimizer(args, model, trainable_params)


    # get ready for distributed training
    if not args.single_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    if 'galore' in args.optimizer.lower(): logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################

    # accumulator for micro (per-microstep) loss used for GA averaging/logging
    micro_loss_sum = 0.0

    for batch_idx, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1

        if update_step >= args.num_training_steps:
            logger.info(f"Reached max number of update steps ({args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        # accumulate raw microstep loss for GA averaging (for logging)
        micro_loss_sum += loss.item()

        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        # At update boundary: compute averaged update loss
        update_loss = micro_loss_sum / float(args.gradient_accumulation)
        micro_loss_sum = 0.0

        # compute and optionally clip grad norm (returns norm)
        if args.grad_clipping != 0.0:
            try:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping))
            except Exception:
                grad_norm = None
        else:
            # compute norm without clipping
            total_norm = 0.0
            for p in trainable_params:
                if p.grad is not None:
                    try:
                        param_norm = p.grad.data.norm(2).item()
                    except Exception:
                        param_norm = 0.0
                    total_norm += param_norm ** 2
            grad_norm = total_norm ** 0.5

        if global_rank == 0:
            pbar.update(1)

        if not layer_wise_flag:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time
        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            getattr(model, "module", model).save_pretrained(current_model_directory, max_shard_size='100GB')

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict() if not layer_wise_flag else {id(k): v.state_dict() for k, v in optimizer_dict.items()},
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                args, model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            if global_rank == 0:
                wandb.log({
                    "update_step": update_step,
                    "eval/loss": total_loss,
                    "eval/ppl": float(np.exp(min(20, total_loss))),
                    "eval/tokens": evaluated_on_tokens,
                    },
                    step=update_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            throughput_examples = (
                args.total_batch_size if args.total_batch_size is not None else args.batch_size * args.gradient_accumulation * world_size
            ) / max(1e-9, update_time)
            # log namespaced metrics at update granularity
            if update_step % LOG_EVERY == 0:
                wandb.log({
                    "update_step": update_step,
                    "train/loss": float(update_loss),
                    "train/ppl": float(np.exp(min(20, update_loss))),
                    "train/lr": lr,
                    "train/tokens_seen": tokens_seen,
                    "train/throughput_tokens_s": tokens_in_update / max(1e-9, update_time),
                    "train/throughput_examples_s": throughput_examples,
                    "train/throughput_batches_s": batches_in_update / max(1e-9, update_time),
                    "train/grad_norm": grad_norm,
                    "sys/gpu_mem_alloc_MB": torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None,
                    "sys/gpu_mem_reserved_MB": torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else None,
                    "sys/gpu_max_alloc_MB": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None,
                    },
                    step=update_step,
                )
        update_time = time.time()
    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        getattr(model, "module", model).save_pretrained(current_model_directory, max_shard_size='100GB')

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict() if not layer_wise_flag else {id(k): v.state_dict() for k, v in optimizer_dict.items()},
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        args, model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
    )

    if global_rank == 0:
        wandb.log({
            "update_step": update_step,
            "eval/loss": total_loss,
            "eval/ppl": float(np.exp(min(20, total_loss))),
            "eval/tokens": evaluated_on_tokens,
            },
            step=update_step,
        )
        logger.info(f"Final eval loss: {total_loss}")

        # tidy finish: add run summary and close
        try:
            if wandb.run is not None:
                wandb.summary["final/update_step"] = update_step
                wandb.summary["final/train/tokens_seen"] = tokens_seen
        except Exception:
            pass
        try:
            wandb.finish()
        except Exception:
            pass

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)