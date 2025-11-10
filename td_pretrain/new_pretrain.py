import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Make Hub calls more tolerant
os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")        # default is 10
os.environ.setdefault("HF_HUB_CONNECTION_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_HTTP_ERROR_RETRIES", "10")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")   # faster if hf_transfer installed
# Optional: reduce metadata thread-fanout so we don't hammer a slow network
os.environ.setdefault("HF_HUB_ENABLE_TELEMETRY", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import time
import json
import random
import argparse
import numpy as np
import math

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
from typing import Tuple
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
# from peft_pretraining.modeling_llama import LlamaForCausalLM

import bitsandbytes as bnb
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor

transformers.logging.set_verbosity_error()

def save_model_and_tokenizer_checkpoint(
    args,
    model,
    tokenizer,
    run_config,
    optimizer,
    scheduler,
    update_step,
    global_step,
    tokens_seen,
    tokens_seen_before,
    update_time,
    layer_wise_flag=False,
    optimizer_dict=None,
    wandb_run=None,
    skip_if_exists=False,
):
    """Save a checkpoint containing model, tokenizer, optimizer, and training state.

    Creates a directory at f"{args.save_dir}/model_{update_step}" and writes:
      - model weights/config via save_pretrained
      - tokenizer files via tokenizer.save_pretrained
      - optimizer.pt (optimizer + optional scheduler state)
      - training_state.json (global/update steps, token counters, update time)
      - wandb.json with run id (if available)

    If skip_if_exists is True and the target directory exists, this is a no-op.
    Returns the path of the checkpoint directory.
    """
    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if skip_if_exists and os.path.exists(current_model_directory):
        return current_model_directory

    os.makedirs(args.save_dir, exist_ok=True)

    # Save model
    getattr(model, "module", model).save_pretrained(current_model_directory, max_shard_size='100GB')

    # Save tokenizer
    try:
        tokenizer.save_pretrained(current_model_directory)
    except Exception as e:
        logger.warning(f"Failed to save tokenizer: {e}")

    # Save q_head if present (sidecar) — HF save_pretrained won't persist ad-hoc modules
    try:
        base_model = getattr(model, "module", model)
        if hasattr(base_model, "q_head") and isinstance(base_model.q_head, nn.Module):
            tmp_q = f"{current_model_directory}/q_head.pt.tmp"
            os.makedirs(current_model_directory, exist_ok=True)
            torch.save(base_model.q_head.state_dict(), tmp_q)
            os.replace(tmp_q, f"{current_model_directory}/q_head.pt")
    except Exception as e:
        logger.warning(f"Failed to save q_head: {e}")

    # Save optimizer + scheduler state
    try:
        if not layer_wise_flag:
            opt_state = optimizer.state_dict()
        else:
            # Build a stable mapping from parameter *names* -> optimizer state_dict
            # Use the module's named_parameters() to find parameter names. If a name
            # cannot be found for a parameter object, fall back to its id (for
            # backward compatibility with older checkpoints).
            opt_state = {}
            base_model = getattr(model, "module", model)
            name_by_id = {id(p): n for n, p in base_model.named_parameters()}
            for p_obj, opt_obj in (optimizer_dict or {}).items():
                p_id = id(p_obj)
                name = name_by_id.get(p_id)
                try:
                    state = opt_obj.state_dict()
                except Exception:
                    state = None
                if name is not None:
                    opt_state[name] = state
                else:
                    # fallback to id-based key for compatibility
                    opt_state[str(p_id)] = state
    except Exception:
        opt_state = None

    optimizer_checkpoint = {
        "optimizer": opt_state,
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "update_step": update_step,
        "global_step": global_step,
        "config": run_config,
        "wandb": getattr(wandb_run, "dir", None) if wandb_run is not None else getattr(getattr(wandb, "run", None), "dir", None),
        "dtype": getattr(args, "dtype", None),
    }
    try:
        tmp_opt = f"{current_model_directory}/optimizer.pt.tmp"
        os.makedirs(current_model_directory, exist_ok=True)
        torch.save(optimizer_checkpoint, tmp_opt)
        os.replace(tmp_opt, f"{current_model_directory}/optimizer.pt")
    except Exception as e:
        logger.warning(f"Failed to save optimizer checkpoint: {e}")

    # Save training state (small JSON) and optional RNG sidecar (binary)
    training_state_checkpoint = {
        "global_step": global_step,
        "update_step": update_step,
        "tokens_seen": tokens_seen,
        "tokens_seen_before": tokens_seen_before,
        "update_time": update_time,
    }
    try:
        tmp_ts = f"{current_model_directory}/training_state.json.tmp"
        os.makedirs(current_model_directory, exist_ok=True)
        with open(tmp_ts, "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)
        os.replace(tmp_ts, f"{current_model_directory}/training_state.json")
    except Exception as e:
        logger.warning(f"Failed to write training_state.json: {e}")

    # Optionally save RNG state as a compact binary sidecar for faster IO
    try:
        if getattr(args, "restore_rng", False):
            rng_state = {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            }
            tmp_rng = f"{current_model_directory}/rng_state.pt.tmp"
            torch.save(rng_state, tmp_rng)
            os.replace(tmp_rng, f"{current_model_directory}/rng_state.pt")
    except Exception as e:
        # Non-critical: RNG save failure shouldn't block checkpointing
        logger.warning(f"Failed to write rng_state sidecar: {e}")

    # Save wandb id at root save_dir for convenience
    try:
        _wandb_id = getattr(wandb_run, "id", None) if wandb_run is not None else getattr(getattr(wandb, "run", None), "id", None)
        if _wandb_id is not None:
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump({"wandb_id": _wandb_id}, f, indent=4)
    except Exception:
        pass

    return current_model_directory

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

    def int_or_none(x):
        return None if x.lower() == "none" else int(x)
    
    def str2bool(x):
        """Parse a string/bool-like value into a proper boolean.

        Accepts: 1/0, true/false, t/f, yes/no, y/n (case-insensitive).
        This avoids argparse's "type=bool" pitfall where non-empty strings are True.
        """
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        return s in {"1", "true", "t", "yes", "y"}

    def str_or_none(x: str | None):
        if x is None:
            return None
        s = str(x).strip().lower()
        return None if s in {"", "none", "null"} else x

    # --- model / training ---
    parser.add_argument("--model_config", type=str, default="meta-llama/Llama-2-7b", required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str_or_none, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", type=str2bool, default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)

    # ---- Absorbing-failure Q-regression knobs ----
    parser.add_argument("--objective", type=str, default="nll",
                        choices=["nll", "q_reg"],
                        help="nll = standard cross-entropy; q_reg = absorbing-failure Q regression")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Discount for Q targets (<= 1). gamma=1 -> remaining token count")
    parser.add_argument("--mix_q_ce", type=float, default=0.0,
                        help="When >0, use total = mix_q_ce * L_q + (1 - mix_q_ce) * L_ce")
    parser.add_argument("--q_neg_samples", type=int, default=0,
                        help="Negatives to sample per (B,T). 0 = full vocab regression")
    parser.add_argument("--use_q_head", type=str2bool, default=False,
                        help="Add a separate linear head to predict Q(s,a) from last hidden states")
    parser.add_argument("--ce_head_only", type=str2bool, default=False,
                        help="When objective=='q_reg', compute CE with hidden DETACHED so only lm_head updates")

    # primary stop condition (kept for backward compatibility)
    parser.add_argument("--num_training_steps", type=int_or_none, default=None,
                        help="Number of **update steps** to train for. Gradient accumulation is already accounted for.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Train until this many tokens are seen (overrides num_training_steps). Suffixes M/B accepted.")

    # NEW: epoch-style controls
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="(Non-streaming only) Train for this many true epochs over the dataset.")
    parser.add_argument("--tokens_per_epoch", type=training_utils.max_train_tokens_to_number, default=None,
                        help="(Streaming or non-streaming) Define a virtual epoch by this many seen tokens.")
    parser.add_argument("--eval_each_epoch", action="store_true", default=False,
                        help="Run eval whenever a (true/virtual) epoch boundary is reached.")
    parser.add_argument("--save_each_epoch", action="store_true", default=False,
                        help="Save a checkpoint whenever a (true/virtual) epoch boundary is reached.")

    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--tags", type=str, default=None)

    # W&B
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
    parser.add_argument("--restore_rng", type=str2bool, default=True,
                        help="If set, save and restore RNG states (torch/cuda/numpy) when checkpointing")
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)   # adafactor beta1 or sgd momentum
    parser.add_argument("--beta2", type=float, default=0.95)  # adamW beta2
    parser.add_argument("--eps", type=float, default=1e-5)

    # (re)init options
    parser.add_argument("--reinit_params", type=str2bool, default=False)
    parser.add_argument("--reinit_scope", type=str, default="all", choices=["all", "lm_head", "embeddings"])
    parser.add_argument("--reinit_seed", type=int, default=None)

    # GaLore
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")

    # runtime
    parser.add_argument("--single_gpu", default=False, action="store_true")

    # --- DATA OPTIONS ---
    # HF remote dataset (default)
    parser.add_argument("--dataset", type=str, default="allenai/c4")
    parser.add_argument("--dataset_config", type=str, default="en",
                        help="Config/name for datasets.load_dataset. Use empty string for none.")

    # NEW: local finite dataset support (true epochs)
    parser.add_argument("--streaming", dest="streaming", action="store_true", default=True,
                        help="Use streaming=True (infinite/unknown length).")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false",
                        help="Use non-streaming finite datasets (true epochs).")
    parser.add_argument("--dataset_local_path", type=str, default=None,
                        help="Path to a datasets *load_from_disk* dataset directory for local loading.")
    parser.add_argument("--train_split", type=str, default="train",
                        help="Train split name when using non-streaming datasets.")
    parser.add_argument("--val_split", type=str, default="validation",
                        help="Validation split name when using non-streaming or local datasets.")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Name of the text field when mapping/tokenizing non-streaming datasets.")

    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(args, model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    was_training = model.training
    model.eval()
    try:
        _time = time.time()

        if args.streaming:
            val_config = args.dataset_config or None
            val_data = datasets.load_dataset(args.dataset, val_config, split="validation", streaming=True, trust_remote_code=True)
            val_data = val_data.shuffle(seed=args.seed)
            # If distributed, split by node first, then create a mapped iterable
            if not args.single_gpu:
                val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

            # Make remove_columns robust by intersecting with present features
            present = set(getattr(val_data, "features", {}) or [])
            to_remove = {"text", "timestamp", "url"} if args.text_column == "text" else {args.text_column}
            remove_cols = list(present.intersection(to_remove))

            # Always map the streaming dataset (single-gpu or distributed) so downstream code
            # can uniformly use val_data_mapped.batch(...)
            val_data_mapped = val_data.map(
                preprocess_batched,
                batched=True,
                remove_columns=remove_cols,
            )

            # batch() helper for streaming
            # Call the helper directly instead of monkey-patching an attribute that
            # may be invoked with keyword args. This avoids lambdas that only accept
            # a positional parameter and prevents TypeError when called with
            # batch_size=... elsewhere.
            iterator = training_utils.batch_fn(val_data_mapped, batch_size)

        else:
            # non-streaming: finite dataset
            if args.dataset_local_path is not None:
                ds = datasets.load_from_disk(args.dataset_local_path)
            else:
                ds = datasets.load_dataset(args.dataset, args.dataset_config or None, split=None, streaming=False, trust_remote_code=True)

            if isinstance(ds, dict):
                if args.val_split in ds:
                    val_raw = ds[args.val_split]
                elif args.train_split in ds:
                    # fallback: small slice of train
                    n = min(10_000, len(ds[args.train_split]))
                    val_raw = ds[args.train_split].select(range(n))
                else:
                    raise ValueError("Validation split not found and no train split to sample from.")
            else:
                # if user directly passed split=... earlier, ds is a Dataset
                val_raw = ds

            # tokenize + keep only model inputs
            # Reuse the tokenizer/preprocess function created for training (do not recreate a new tokenizer per batch)
            # Compute remove_columns robustly to avoid depending on any fixed dataset schema
            present = set(getattr(val_raw, "features", {}) or [])
            remove_cols = list(present.difference({args.text_column}))
            val_tok = val_raw.map(
                lambda batch: preprocess_batched(batch),
                batched=True,
                remove_columns=remove_cols,
            )
            val_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

            if not args.single_gpu:
                sampler = torch.utils.data.distributed.DistributedSampler(val_tok, shuffle=False)
            else:
                sampler = None

            iterator = torch.utils.data.DataLoader(
                val_tok, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=min(4, args.workers), pin_memory=True
            )

        logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

        target_eval_tokens = 4096
        evaluated_on_tokens = 0
        total_loss = torch.tensor(0.0, device=device)
        total_batches = 0

        for batch in iterator:
            if evaluated_on_tokens > target_eval_tokens:
                break
            total_batches += 1

            # streaming iterator yields dict of tensors already; map-style yields dict from DataLoader
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            loss = model(**batch, labels=labels).loss
            total_loss += loss.detach()

            evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

        total_loss = total_loss / max(1, total_batches)

        # reduce across ranks
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


def _default_module_init_(module, std=0.02):
    # Conservative, HF-like defaults for Linear/Embedding/LayerNorm
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _reinit_linear_(linear, std):
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _reinit_embeddings_(emb, std):
    nn.init.normal_(emb.weight, mean=0.0, std=std)


def reinitialize_model_parameters(model, scope="all", seed=None):
    """
    Reinitialize HF model params.
    scope: "all" | "lm_head" | "embeddings"
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Try to respect the model's preferred init std if present
    init_std = getattr(getattr(model, "config", None), "initializer_range", 0.02)

    if scope == "all":
        # Prefer the model's built-in init if available
        if hasattr(model, "init_weights") and callable(getattr(model, "init_weights")):
            try:
                model.init_weights()
                return
            except Exception:
                pass
        if hasattr(model, "_init_weights") and callable(getattr(model, "_init_weights")):
            try:
                model.apply(lambda m: model._init_weights(m))  # type: ignore[attr-defined]
                return
            except Exception:
                pass

        model.apply(lambda m: _default_module_init_(m, std=init_std))

        # Some models need this to ensure tied weights remain consistent
        if hasattr(model, "tie_weights") and callable(getattr(model, "tie_weights")):
            try:
                model.tie_weights()
            except Exception:
                pass
        return

    if scope == "lm_head":
        head = getattr(model, "lm_head", None)
        if isinstance(head, nn.Linear):
            _reinit_linear_(head, std=init_std)
        # If the model ties embeddings & head, re-tie after touching the head
        if hasattr(model, "tie_weights") and callable(getattr(model, "tie_weights")):
            try:
                model.tie_weights()
            except Exception:
                pass
        return

    if scope == "embeddings":
        # input embeddings
        emb_in = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if isinstance(emb_in, nn.Embedding):
            _reinit_embeddings_(emb_in, std=init_std)
        # output head sometimes acts as output embeddings; refresh + tie if applicable
        head = getattr(model, "lm_head", None)
        if isinstance(head, nn.Linear):
            _reinit_linear_(head, std=init_std)
        if hasattr(model, "tie_weights") and callable(getattr(model, "tie_weights")):
            try:
                model.tie_weights()
            except Exception:
                pass
        return


def ddp_token_weighted_avg(sum_loss_local, sum_tokens_local, device, single_gpu):
    """Return the token-weighted average loss across DDP ranks.

    sum_loss_local: scalar (float) - sum of per-example losses weighted by token count on this rank
    sum_tokens_local: scalar (float) - total non-pad tokens on this rank
    device: device string (e.g., 'cuda:0') used to create reduction tensor
    single_gpu: bool, if True no distributed reduce is performed
    """
    vec = torch.tensor(
        [float(sum_loss_local), float(sum_tokens_local)],
        device=device, dtype=torch.float32
    )
    if (not single_gpu) and dist.is_available() and dist.is_initialized():
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    sum_loss_all, sum_tokens_all = vec[0].item(), max(1.0, vec[1].item())
    return sum_loss_all / sum_tokens_all


def _geometric_series_steps(steps: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    steps: non-negative integer tensor K (remaining valid steps, BxT).
    Return sum_{i=0}^{K-1} gamma^i. Vectorized & numerically stable for gamma=1.
    """
    if float(gamma) == 1.0:
        return steps.to(dtype=torch.float32)
    # (1 - gamma^K) / (1 - gamma)
    g = torch.tensor(gamma, device=steps.device, dtype=torch.float32)
    return (1.0 - torch.pow(g, steps.to(torch.float32))) / (1.0 - g)


@torch.no_grad()
def _build_absorbing_failure_V_targets_from_labels(labels: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    labels: (B,T) with -100 on ignored positions (pads/etc).
    V[b,t] = sum_{k=t}^{L-1} gamma^{k-t}, where L=#valid tokens in that row.
    """
    valid = (labels != -100)                        # (B,T) bool
    B, T = labels.shape
    lengths = valid.sum(dim=1)                      # (B,)
    arange_T = torch.arange(T, device=labels.device).unsqueeze(0).expand(B, T)
    steps_remaining = (lengths.unsqueeze(1) - arange_T).clamp_min(0)  # (B,T)
    V = _geometric_series_steps(steps_remaining, gamma)               # (B,T)
    V = V * valid.to(V.dtype)                                         # zero out invalid
    return V


def _q_regression_loss(
    hidden_last: torch.Tensor,            # (B,T,H)
    logits: torch.Tensor,                 # (B,T,V)
    labels: torch.Tensor,                 # (B,T) with -100 ignored
    gamma: float,
    q_neg_samples: int,
    use_q_head: bool,
    base_model: torch.nn.Module,
):
    """
    Returns (loss_q, q_pred). If use_q_head, predicts with base_model.q_head(hidden_last);
    otherwise reuse logits as Q(s,a).
    Targets: Q*(s_t, a_t^*) = V_targets[b,t], Q*(s_t, a!=a_t^*) = 0.
    """
    B, T, V = logits.shape
    if use_q_head and hasattr(base_model, "q_head"):
        # Ensure hidden_last dtype matches q_head weights to avoid bf16/float32 matmul errors
        q_head = base_model.q_head
        try:
            if hidden_last.dtype != q_head.weight.dtype:
                hidden_last = hidden_last.to(q_head.weight.dtype)
        except Exception:
            # Be defensive: if q_head has no weight attr or cast fails, continue and let downstream raise
            pass
        q_pred = q_head(hidden_last)     # (B,T,V)
    else:
        q_pred = logits

    # --- Build V targets and ensure they match q_pred dtype/device ---
    V_targets = _build_absorbing_failure_V_targets_from_labels(labels, gamma=gamma)  # (B,T) on labels' device, float32
    # Cast/move value targets to q_pred's dtype and device to avoid scatter/gather dtype mismatches
    V_targets = V_targets.to(dtype=q_pred.dtype, device=q_pred.device)
    V_targets_exp = V_targets.unsqueeze(-1)                                          # (B,T,1), same dtype/device as q_pred

    if q_neg_samples and q_neg_samples > 0:
        gt_idx = labels.clamp_min(0).unsqueeze(-1)           # (B,T,1); clamp avoids -100 gather errors
        K = int(q_neg_samples)
        neg_idx = torch.randint(0, V, (B, T, K), device=labels.device)
        idxs = torch.cat([gt_idx, neg_idx], dim=-1)          # (B,T,K+1)

        # Move indices to q_pred device before gather
        if idxs.device != q_pred.device:
            idxs = idxs.to(q_pred.device)
        q_slice = q_pred.gather(dim=-1, index=idxs)          # (B,T,K+1)

        y_slice = torch.zeros_like(q_slice)
        mask_valid = (labels != -100).unsqueeze(-1)
        # Ensure mask and V_targets live on q_pred device/dtype for safe assignment
        mask_valid_dev = mask_valid.to(q_pred.device)
        # y0 has same dtype as V_targets (already casted); cast to y_slice dtype when assigning
        y0 = torch.where(mask_valid_dev.squeeze(-1), V_targets, torch.zeros_like(V_targets))
        y_slice[..., 0] = y0.to(y_slice.dtype)
        loss_q = 0.5 * (q_slice - y_slice).pow(2).mean()
    else:
        Q_targets = torch.zeros_like(q_pred)
        mask_valid = (labels != -100)
        if mask_valid.any():
            # indices must live on q_pred.device for scatter_
            idx = labels.clamp_min(0).unsqueeze(-1)
            if idx.device != q_pred.device:
                idx = idx.to(q_pred.device)
            # src (V_targets_exp) already matches q_pred dtype/device
            Q_targets.scatter_(dim=-1, index=idx, src=V_targets_exp)
            Q_targets = Q_targets * mask_valid.to(Q_targets.device).unsqueeze(-1).to(Q_targets.dtype)
        loss_q = 0.5 * (q_pred - Q_targets).pow(2).mean()

    return loss_q, q_pred


def compute_loss(
    args,
    model: torch.nn.Module,
    batch: dict,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Centralized loss: supports 'nll' and 'q_reg'.

    Returns:
        total_loss: torch scalar
        parts: dict with optional keys:
            - 'q_loss': scalar tensor (only when objective=='q_reg')
            - 'ntp_loss': scalar tensor (next-token CE; when objective=='q_reg')
    """
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    base_model = model.module if is_ddp else model

    if args.objective == "nll":
        out = model(**batch, labels=labels)
        return out.loss, {}

    if args.objective == "q_reg":
        out = model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)
        logits = out.logits
        hidden_last = out.hidden_states[-1]  # (B,T,H)

        # Q regression loss
        loss_q, _ = _q_regression_loss(
            hidden_last=hidden_last,
            logits=logits,
            labels=labels,
            gamma=args.gamma,
            q_neg_samples=args.q_neg_samples,
            use_q_head=args.use_q_head,
            base_model=base_model,
        )

        # --- Always compute CE (NTP) for logging, even if mix_q_ce == 0 ---
        if args.ce_head_only:
            # head-only CE: run detached last hidden through final norm (if present)
            # then through lm_head so only lm_head (and any head-only params) get gradients.
            lm_head = base_model.get_output_embeddings()   # nn.Linear-like

            # Qwen2 layout: final norm lives at base_model.model.norm (fall back to base_model.norm)
            final_norm = getattr(getattr(base_model, "model", base_model), "norm", None)

            hl = hidden_last.detach()
            if final_norm is not None:
                # apply final norm without grad to avoid changing base model
                hl = final_norm(hl)

            # ensure dtype matches lm_head weights (important for bf16/fp16)
            lm_w = getattr(lm_head, "weight", None)
            if lm_w is not None and hl.dtype != lm_w.dtype:
                hl = hl.to(lm_w.dtype)

            logits_ce = lm_head(hl)
        else:
            logits_ce = logits

        # causal shift: next-token CE compares logits[:, :-1] with labels[:, 1:]
        logits_shift = logits_ce[:, :-1, :].contiguous()
        labels_shift = labels[:, 1:].contiguous()

        loss_ce = torch.nn.functional.cross_entropy(
            logits_shift.view(-1, logits_shift.size(-1)),
            labels_shift.view(-1),
            ignore_index=-100,
        )

        # Mixture for the optimization objective
        if args.mix_q_ce > 0.0:
            if args.ce_head_only:
                total = loss_q + loss_ce
            else:
                total = args.mix_q_ce * loss_q + (1.0 - args.mix_q_ce) * loss_ce   
            
        else:
            total = loss_q

        return total, {"q_loss": loss_q, "ntp_loss": loss_ce}

    raise ValueError(f"Unknown objective: {args.objective}")

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
    tokenizer = AutoTokenizer.from_pretrained(args.model_config, model_max_length=args.max_length)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    tokenizer.padding_side = "right"

    def preprocess_batched(batch):
        return tokenizer(
            batch[args.text_column] if args.text_column in batch else batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    if args.streaming:
        data_config = args.dataset_config or None
        data = datasets.load_dataset(args.dataset, data_config, split="train", streaming=True, trust_remote_code=True)
        logger.info(f"Shuffling data with seed {args.seed}")
        data = data.shuffle(seed=args.seed)

        if not args.single_gpu:
            data = datasets.distributed.split_dataset_by_node(data, rank=global_rank, world_size=world_size)

        dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        train_num_examples = None  # unknown
        return tokenizer, preprocess_batched, dataloader, train_num_examples

    # ---- Non-streaming: finite dataset (true epochs) ----
    if args.dataset_local_path is not None:
        ds = datasets.load_from_disk(args.dataset_local_path)
    else:
        ds = datasets.load_dataset(args.dataset, args.dataset_config or None, split=None, streaming=False, trust_remote_code=True)

    if isinstance(ds, dict):
        if args.train_split not in ds:
            raise ValueError(f"train split '{args.train_split}' not found in dataset")
        train_raw = ds[args.train_split]
    else:
        train_raw = ds  # user passed a concrete split elsewhere

    train_tok = train_raw.map(
        lambda batch: tokenizer(
            batch[args.text_column], max_length=args.max_length, truncation=True, padding="max_length"
        ),
        batched=True,
        remove_columns=[c for c in train_raw.column_names if c != args.text_column],
    )
    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

    if not args.single_gpu:
        sampler = torch.utils.data.distributed.DistributedSampler(train_tok, shuffle=True)
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        train_tok,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    train_num_examples = len(train_tok)
    return tokenizer, preprocess_batched, dataloader, train_num_examples

def load_model(args, device):

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    update_time = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        torch_dtype = torch.bfloat16 if args.dtype in ["bf16", "bfloat16"] else None
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            args.continue_from, torch_dtype=torch_dtype
        )
        model_config = AutoConfig.from_pretrained(args.continue_from)

        if (not args.reinit_params) and os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            # Only load training state if we're *not* reinitializing
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state.get("global_step", 0)
            update_step = _old_state.get("update_step", 0)
            tokens_seen = _old_state.get("tokens_seen", 0)
            tokens_seen_before = _old_state.get("tokens_seen_before", 0)
            # restore update_time from checkpoint so we can keep throughput calcs continuous
            update_time = _old_state.get("update_time", 0)
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
            # Clear resume summary message for human readability
            logger.info(f"Resuming training from update_step={update_step}, global_step={global_step}")
            # Optionally restore RNGs (torch/cuda/numpy) if present and requested
            try:
                if getattr(args, "restore_rng", False):
                    if "torch_rng_state" in _old_state:
                        try:
                            torch.set_rng_state(torch.tensor(_old_state["torch_rng_state"], dtype=torch.uint8))
                        except Exception:
                            pass
                    if torch.cuda.is_available() and _old_state.get("cuda_rng_state") is not None:
                        try:
                            for i, s in enumerate(_old_state.get("cuda_rng_state", [])):
                                torch.cuda.set_rng_state(torch.tensor(s, dtype=torch.uint8), device=i)
                        except Exception:
                            pass
                    if _old_state.get("numpy_rng_state") is not None:
                        try:
                            np_state = _old_state["numpy_rng_state"]
                            np.random.set_state((np_state.get("name"), np.array(np_state.get("state"), dtype=np.int64), int(np_state.get("pos"))))
                        except Exception:
                            pass
            except Exception:
                logger.warning("Failed to restore RNG state from checkpoint (continuing without restoring RNGs)")
        else:
            if args.reinit_params:
                logger.info("Ignoring saved training_state.* because --reinit_params is set (fresh training).")
            else:
                logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)
    else:
        model_config = AutoConfig.from_pretrained(args.model_config)
        if args.use_hf_model:
            model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
        else:
            model = AutoModelForCausalLM.from_config(model_config)

    # >>> NEW: Reinitialize if requested (do this BEFORE moving to device/dtype for speed/clarity)
    if args.reinit_params:
        try:
            logger.info(f"Reinitializing model parameters (scope='{args.reinit_scope}')")
            reinitialize_model_parameters(model, scope=args.reinit_scope, seed=args.reinit_seed)
        except Exception as e:
            logger.warning(f"Reinitialization failed: {e}")

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    return model, model_config, global_step, update_step, beginning_step, tokens_seen, tokens_seen_before, update_time

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
        # Show absolute update steps and start the bar at the restored update_step
        # so the UI matches the resumed training state.
        try:
            pbar = tqdm(
                total=args.num_training_steps,
                initial=update_step,
                desc="Update steps",
                ncols=80,
            )
        except Exception:
            # Fall back to previous behavior if something unexpected occurs
            pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    return trainable_params, run_config, pbar


def init_wandb_once(args, run_config, global_rank):
    """Initialize wandb one time on rank 0 and pass the full run_config.
    Returns the wandb Run object or None for non-rank0 processes."""
    if global_rank != 0:
        return None

    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None

    # Choose resume behavior: if a wandb_id was provided we must attach to the
    # same run (fail-fast if it doesn't exist). Otherwise allow creating a new run.
    _resume = "must" if args.wandb_id else "allow"

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
        resume=_resume,
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

    # If we resumed an existing run, allow config values to be updated without errors
    try:
        if args.wandb_id:
            wandb.config.update(run_config, allow_val_change=True)
    except Exception:
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
    tokenizer, preprocess_batched, dataloader, train_num_examples = tokenizer_data_loader(args, global_rank, world_size)
    model, model_config, global_step, update_step, beginning_step, tokens_seen, tokens_seen_before, saved_update_time = load_model(args, device)
    # Ensure model embeddings match tokenizer vocabulary if we added a pad token
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        logger.warning(f"resize_token_embeddings call failed or unnecessary: {e}")

    # If user requested embedding-only reinit, do it AFTER resize so shapes match and
    # newly added tokens are initialized as well.
    try:
        if args.reinit_params and args.reinit_scope == "embeddings":
            logger.info("Reinitializing embeddings after resize_token_embeddings")
            reinitialize_model_parameters(model, scope="embeddings", seed=args.reinit_seed)
    except Exception as e:
        logger.warning(f"Reinitializing embeddings after resize failed: {e}")

    # ---- Q head (optional): predict Q(s,a) from last hidden state ----
    try:
        if args.use_q_head:
            # infer hidden size from common config fields
            raw_hidden = getattr(getattr(model, "config", None), "hidden_size", None) or \
                         getattr(getattr(model, "config", None), "n_embd", None) or \
                         getattr(getattr(model, "config", None), "hidden_dim", None)
            if raw_hidden is None:
                raise ValueError("Could not infer hidden size for q_head; expected model.config.hidden_size (or n_embd/hidden_dim)")
            hidden_size = int(raw_hidden)

            # infer vocab size robustly from model output head
            out_emb = model.get_output_embeddings()
            if hasattr(out_emb, "num_embeddings"):
                vocab_size = int(out_emb.num_embeddings)
            elif hasattr(out_emb, "out_features"):
                vocab_size = int(out_emb.out_features)
            else:
                raise ValueError("Could not infer vocab size from model.get_output_embeddings()")

            # Attach a simple linear head. Construct it on the same device & dtype as the model
            model_dtype = next(model.parameters()).dtype
            model_device = next(model.parameters()).device
            model.q_head = nn.Linear(hidden_size, vocab_size, bias=True).to(device=model_device, dtype=model_dtype)
            logger.info(f"Attached q_head on model with hidden_size={hidden_size}, vocab_size={vocab_size}, dtype={model_dtype}, device={model_device}")

            # If resuming from a checkpoint directory, try to restore q_head state_dict from a q_head.pt sidecar
            try:
                if args.continue_from is not None:
                    q_head_ckpt = os.path.join(args.continue_from, "q_head.pt")
                    if os.path.exists(q_head_ckpt):
                        sd = torch.load(q_head_ckpt, map_location="cpu")
                        base_model = getattr(model, "module", model)
                        if hasattr(base_model, "q_head") and isinstance(base_model.q_head, nn.Module):
                            base_model.q_head.load_state_dict(sd, strict=True)
                            # ensure q_head is on correct device/dtype after loading
                            base_model.q_head.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
                            logger.info("Loaded q_head from checkpoint")
                    else:
                        logger.warning("q_head.pt not found in resume dir; q_head will start fresh")
            except Exception as e:
                logger.warning(f"Could not load q_head from checkpoint: {e}")
    except Exception as e:
        logger.warning(f"Failed to attach q_head: {e}")

    # --- Derive an effective num_training_steps for scheduler/pbar ---
    # Priority order:
    # 1) max_train_tokens (if given) -> converted to steps online (no change here; we keep your existing logic)
    # 2) Non-streaming + num_epochs (true epochs) -> compute updates from dataset size
    # 3) Otherwise, use user-provided num_training_steps
    if (not args.streaming) and (args.num_epochs is not None) and (train_num_examples is not None):
        per_update_examples = args.batch_size * args.gradient_accumulation * world_size
        updates_per_epoch = math.ceil(train_num_examples / max(1, per_update_examples))
        args.num_training_steps = updates_per_epoch * args.num_epochs
        logger.info(f"[epochs] train examples={train_num_examples}, per_update_examples={per_update_examples}, "
                    f"updates/epoch={updates_per_epoch}, total updates={args.num_training_steps}")

    # If max_train_tokens is provided, derive num_training_steps from token cap so pbar/scheduler reflect token-based stopping
    if args.max_train_tokens is not None:
        per_update_tokens = args.max_length * args.batch_size * args.gradient_accumulation * world_size
        per_update_tokens = max(1, per_update_tokens)
        args.num_training_steps = math.ceil(args.max_train_tokens / per_update_tokens)
        logger.info(f"[tokens cap] per_update_tokens≈{per_update_tokens}, derived total updates={args.num_training_steps}")

    # If user requested resume but there's no training_state.json, warn that progress bar/counts will restart
    if args.continue_from and not os.path.exists(os.path.join(args.continue_from, "training_state.json")):
        logger.warning("No training_state.json found in continue_from; progress bar and update_step will start from 0 unless training_state.json is present.")

    trainable_params, run_config, pbar = get_param_progress_bar(args, model, model_config, device, world_size, global_rank, update_step)

    # Log the final planned number of update steps (after any recomputation)
    if global_rank == 0:
        logger.info(f"Will train for {args.num_training_steps - update_step} update steps")

    # Convenience: if continuing from a checkpoint and the user didn't pass a
    # wandb run id, prefer to auto-load wandb.json from the checkpoint dir.
    if args.wandb_id is None and args.continue_from:
        wandb_hint = os.path.join(args.continue_from, "wandb.json")
        if os.path.exists(wandb_hint):
            try:
                with open(wandb_hint, "r") as _f:
                    w = json.load(_f)
                    if isinstance(w, dict) and w.get("wandb_id"):
                        args.wandb_id = w.get("wandb_id")
                        logger.info(f"Auto-loaded wandb_id from {wandb_hint}")
            except Exception:
                pass

# ----- NEW: force resume behavior via env (harmless if already set) -----
    if args.wandb_id:
        # require attaching to the same run id; fail-fast if the id does not exist
        os.environ.setdefault("WANDB_RESUME", "must")
        os.environ.setdefault("WANDB_RUN_ID", args.wandb_id)
    else:
        # allow creating a fresh run when no id provided
        os.environ.setdefault("WANDB_RESUME", "allow")

    # Streaming resume note
    if args.streaming and args.continue_from:
        logger.info("Streaming mode: resuming counters but not exact data position.")

    # Initialize W&B once (rank 0). We pass the full run_config here.
    wandb_run = init_wandb_once(args, run_config, global_rank)

    # Optional: log a tiny resume marker so the UI shows where a resume occurred
    try:
        if global_rank == 0 and args.continue_from and update_step > 0 and wandb_run is not None:
            wandb.log({"system/resumed": 1}, step=update_step)
    except Exception:
        pass

    optimizer, layer_wise_flag, optimizer_dict, galore_params, scheduler = load_optimizer(args, model, trainable_params)

    # --- Broadcast model params from rank 0 to ensure exact same weights on resume in DDP setups ---
    if (not args.single_gpu) and dist.is_available() and dist.is_initialized():
        try:
            if global_rank == 0:
                logger.info("Broadcasting model parameters from rank 0 to all ranks to ensure parity on resume")
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
        except Exception as e:
            logger.warning(f"Model parameter broadcast failed: {e}")

    # --- Resume optimizer/scheduler if continuing ---
    if args.continue_from is not None:
        ckpt_path = os.path.join(args.continue_from, "optimizer.pt")
        if os.path.exists(ckpt_path):
            try:
                logger.info(f"Loading optimizer/scheduler state from {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location="cpu")
                scheduler_loaded = False

                # Config/schedule drift check
                try:
                    saved_cfg = ckpt.get("config", {}) or {}
                    if saved_cfg:
                        diffs = []
                        for k in ["scheduler", "num_training_steps", "warmup_steps", "min_lr_ratio", "max_lr"]:
                            saved_val = str(saved_cfg.get(k))
                            if k == "max_lr":
                                curr_val = str(run_config.get("max_lr"))
                            else:
                                curr_val = str(getattr(args, k, None))
                            if saved_val != "None" and saved_val != curr_val:
                                diffs.append(k)
                        if diffs:
                            logger.warning(f"Resume config differs for: {', '.join(diffs)}. Learning-rate schedule shape may change.")
                except Exception:
                    pass

                if not layer_wise_flag:
                    if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                        try:
                            optimizer.load_state_dict(ckpt["optimizer"])
                        except Exception as e:
                            logger.warning(f"Could not load optimizer state_dict (non-layerwise): {e}")
                else:
                    # ckpt["optimizer"] expected to be a dict keyed by parameter name (preferred)
                    if "optimizer" in ckpt and isinstance(ckpt["optimizer"], dict):
                        base_model = getattr(model, "module", model)
                        name_to_param = dict(base_model.named_parameters())
                        for k, state in ckpt["optimizer"].items():
                            # If the key is numeric (old style), fall back to id-based matching
                            if isinstance(k, str) and k.isdigit():
                                try:
                                    k_int = int(k)
                                except Exception:
                                    continue
                                for p_obj, opt_obj in optimizer_dict.items():
                                    if id(p_obj) == k_int:
                                        try:
                                            opt_obj.load_state_dict(state)
                                        except Exception:
                                            logger.warning(f"Failed loading layer-wise optimizer state for param id {k_int}")
                                        break
                                continue

                            # Prefer name-based matching
                            p = name_to_param.get(k)
                            if p is None:
                                logger.warning(f"Param {k} missing in current model; skipping its optimizer state")
                                continue
                            opt = optimizer_dict.get(p)
                            if opt is None:
                                logger.warning(f"No optimizer object for parameter {k}; skipping")
                                continue
                            try:
                                opt.load_state_dict(state)
                            except Exception as e:
                                logger.warning(f"Failed loading optimizer state for {k}: {e}")

                if scheduler is not None and "scheduler" in ckpt and ckpt.get("scheduler") is not None:
                    try:
                        scheduler.load_state_dict(ckpt["scheduler"])
                        scheduler_loaded = True
                    except Exception as e:
                        logger.warning(f"Failed to load scheduler state: {e}")

                # If we didn't restore scheduler state but we have an update_step, advance scheduler to match update count
                if scheduler is not None and (not locals().get("scheduler_loaded", False)) and update_step > 0:
                    try:
                        for _ in range(update_step):
                            scheduler.step()
                    except Exception:
                        logger.warning("Failed to advance scheduler by update_step; skipping incremental stepping")

                # Restore RNG sidecar if requested
                try:
                    if getattr(args, "restore_rng", False):
                        rng_pt = os.path.join(args.continue_from, "rng_state.pt")
                        if os.path.exists(rng_pt):
                            try:
                                r = torch.load(rng_pt, map_location="cpu")
                                if r.get("torch_cpu") is not None:
                                    torch.set_rng_state(r["torch_cpu"])
                                if torch.cuda.is_available() and r.get("torch_cuda_all") is not None:
                                    torch.cuda.set_rng_state_all(r["torch_cuda_all"])
                                if r.get("numpy") is not None:
                                    np.random.set_state(r["numpy"])
                                if r.get("python") is not None:
                                    random.setstate(r["python"])
                                logger.info("Restored RNG state from rng_state.pt sidecar")
                            except Exception as e:
                                logger.warning(f"Failed to restore RNG sidecar: {e}")
                except Exception:
                    pass

                logger.info("Loaded optimizer and scheduler state from checkpoint (if present)")
            except Exception as e:
                logger.warning(f"Failed to load optimizer/scheduler state: {e}")

    # Optionally save immediately on resume so the user sees a checkpoint at the
    # restored update_step (only on rank 0). Skip update_step==0 because an
    # initial checkpoint is already saved earlier.
    try:
        if args.continue_from is not None and global_rank == 0 and update_step > 0 and (update_step % args.save_every == 0):
            logger.info(f"Saving checkpoint on resume at update step {update_step}")
            save_model_and_tokenizer_checkpoint(
                args=args,
                model=model,
                tokenizer=tokenizer,
                run_config=run_config,
                optimizer=optimizer,
                scheduler=scheduler,
                update_step=update_step,
                global_step=global_step,
                tokens_seen=tokens_seen,
                tokens_seen_before=tokens_seen_before,
                update_time=update_time,
                layer_wise_flag=layer_wise_flag,
                optimizer_dict=optimizer_dict,
                wandb_run=wandb_run,
            )
    except Exception as e:
        logger.warning(f"Failed to write resume checkpoint: {e}")


    # get ready for distributed training
    if not args.single_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    if 'galore' in args.optimizer.lower(): logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    # Save an initial checkpoint (model + tokenizer + optimizer state) before training starts
    # so inference engines like vLLM can load directly from a checkpoint folder, and for easier debugging.
    if global_rank == 0 and update_step == 0:
        logger.info("Saving initial model/tokenizer/optimizer state (before training)")
        save_model_and_tokenizer_checkpoint(
            args=args,
            model=model,
            tokenizer=tokenizer,
            run_config=run_config,
            optimizer=optimizer,
            scheduler=scheduler,
            update_step=update_step,
            global_step=global_step,
            tokens_seen=tokens_seen,
            tokens_seen_before=tokens_seen_before,
            update_time=0,
            layer_wise_flag=layer_wise_flag,
            optimizer_dict=optimizer_dict,
            wandb_run=wandb_run,
            skip_if_exists=True,
        )

    # -------------------------
    # TRAINING LOOP(S)
    # -------------------------
    pad_idx = tokenizer.pad_token_id
    # Initialize update_time so throughput immediately after resume doesn't spike.
    try:
        if saved_update_time and saved_update_time > 0:
            # Try to initialize so the first measured delta approximates the last saved update_time
            update_time = time.time() - float(saved_update_time)
        else:
            update_time = time.time()
    except Exception:
        update_time = time.time()
    local_step = 0
    micro_loss_sum = 0.0
    epoch_idx = 0
    epoch_start_tokens = 0  # for virtual epochs by tokens

    # helper to run end-of-epoch hooks
    def _maybe_epoch_hooks():
        nonlocal epoch_idx, epoch_start_tokens
        did_epoch = False

        # Virtual epoch by tokens (works in streaming or non-streaming)
        if args.tokens_per_epoch is not None:
            if (tokens_seen - epoch_start_tokens) >= args.tokens_per_epoch:
                epoch_idx += 1
                epoch_start_tokens = tokens_seen
                logger.info(f"Reached end of epoch {epoch_idx} (~{args.tokens_per_epoch} tokens)")
                did_epoch = True

        # True epoch hook: in non-streaming + num_epochs mode, we bump epoch_idx in the outer loop
        # and call hooks at the end of each outer epoch as well (below).
        if did_epoch and args.eval_each_epoch:
            total_loss, evaluated_on_tokens = evaluate_model(
                args, model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            if global_rank == 0:
                wandb.log({
                    "update_step": update_step,
                    "epoch": epoch_idx,
                    "eval/loss": total_loss,
                    "eval/ppl": float(np.exp(min(20, total_loss))),
                    "eval/tokens": evaluated_on_tokens,
                }, step=update_step)

        if did_epoch and args.save_each_epoch and global_rank == 0:
            logger.info(f"Saving checkpoint at end of epoch {epoch_idx}")
            save_model_and_tokenizer_checkpoint(
                args=args,
                model=model,
                tokenizer=tokenizer,
                run_config=run_config,
                optimizer=optimizer,
                scheduler=scheduler,
                update_step=update_step,
                global_step=global_step,
                tokens_seen=tokens_seen,
                tokens_seen_before=tokens_seen_before,
                update_time=update_time,
                layer_wise_flag=layer_wise_flag,
                optimizer_dict=optimizer_dict,
                wandb_run=wandb_run,
            )

    # Choose loop style
    if (not args.streaming) and (args.num_epochs is not None):
        # ---- TRUE EPOCHS ----
        assert hasattr(dataloader, "__iter__"), "Non-streaming dataloader expected to be finite/iterable per epoch."
        for epoch in range(args.num_epochs):
            # important for DDP shuffling
            if (not args.single_gpu) and hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")

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

                # Honor max_train_tokens cap (stop early if reached)
                if args.max_train_tokens is not None and tokens_seen >= args.max_train_tokens:
                    logger.info(f"Reached max_train_tokens={args.max_train_tokens}. Stopping training.")
                    break

                # compute loss (supports nll or q_reg objectives)
                loss, loss_parts = compute_loss(args, model, batch, labels)

                # ---- token-weighted accumulation across GA microsteps ----
                valid_tokens = (labels != -100).sum()
                # accumulate *on CPU as floats* to avoid dtype surprises
                if 'sum_loss_tok' not in locals():
                    sum_loss_tok = 0.0
                    sum_tokens_tok = 0.0
                sum_loss_tok += float(loss.detach()) * float(valid_tokens.item())
                sum_tokens_tok += float(valid_tokens.item())

                # NEW: part-wise accumulators for q_reg
                if args.objective == "q_reg":
                    if 'sum_q_loss_tok' not in locals():
                        sum_q_loss_tok = 0.0
                        sum_ntp_loss_tok = 0.0
                    if "q_loss" in loss_parts and loss_parts["q_loss"] is not None:
                        sum_q_loss_tok += float(loss_parts["q_loss"].detach()) * float(valid_tokens.item())
                    if "ntp_loss" in loss_parts and loss_parts["ntp_loss"] is not None:
                        sum_ntp_loss_tok += float(loss_parts["ntp_loss"].detach()) * float(valid_tokens.item())

                # backprop (still per-microstep)
                scaled_loss = loss / args.gradient_accumulation
                scaled_loss.backward()

                # wait until GA boundary
                if global_step % args.gradient_accumulation != 0:
                    continue

                # ---- form local token-weighted mean, then DDP-reduce to global ----
                update_loss_global = ddp_token_weighted_avg(
                    sum_loss_tok, sum_tokens_tok, device=device, single_gpu=args.single_gpu
                )

                # NEW: part-wise global reduces for q_reg
                if args.objective == "q_reg":
                    q_loss_global = ddp_token_weighted_avg(
                        sum_q_loss_tok, sum_tokens_tok, device=device, single_gpu=args.single_gpu
                    )
                    ntp_loss_global = ddp_token_weighted_avg(
                        sum_ntp_loss_tok, sum_tokens_tok, device=device, single_gpu=args.single_gpu
                    )

                # reset accumulators for next update
                sum_loss_tok = 0.0
                sum_tokens_tok = 0.0
                if args.objective == "q_reg":
                    sum_q_loss_tok = 0.0
                    sum_ntp_loss_tok = 0.0

                if args.grad_clipping != 0.0:
                    try:
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping))
                    except Exception:
                        grad_norm = None
                else:
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

                if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
                    logger.info(f"Saving checkpoint at update step {update_step}")
                    save_model_and_tokenizer_checkpoint(
                        args=args,
                        model=model,
                        tokenizer=tokenizer,
                        run_config=run_config,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        update_step=update_step,
                        global_step=global_step,
                        tokens_seen=tokens_seen,
                        tokens_seen_before=tokens_seen_before,
                        update_time=update_time,
                        layer_wise_flag=layer_wise_flag,
                        optimizer_dict=optimizer_dict,
                        wandb_run=wandb_run,
                    )

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
                        }, step=update_step)
                    logger.info(f"Eval loss at step {update_step}: {total_loss}")

                lr = (optimizer.param_groups[0]["lr"]
                      if not layer_wise_flag else list(optimizer_dict.values())[0].param_groups[0]["lr"])
                tokens_in_update = tokens_seen - tokens_seen_before
                tokens_seen_before = tokens_seen
                batches_in_update = args.gradient_accumulation * world_size

                if global_rank == 0 and update_step % LOG_EVERY == 0:
                    throughput_examples = (
                        args.total_batch_size if args.total_batch_size is not None
                        else args.batch_size * args.gradient_accumulation * world_size
                    ) / max(1e-9, update_time)
                    logdict = {
                        "update_step": update_step,
                        "train/loss": float(update_loss_global),
                        "train/ppl": float(np.exp(min(20, update_loss_global))),
                        "train/lr": lr,
                        "train/tokens_seen": tokens_seen,
                        "train/throughput_tokens_s": tokens_in_update / max(1e-9, update_time),
                        "train/throughput_examples_s": throughput_examples,
                        "train/throughput_batches_s": batches_in_update / max(1e-9, update_time),
                        "train/grad_norm": grad_norm,
                        "sys/gpu_mem_alloc_MB": torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None,
                        "sys/gpu_mem_reserved_MB": torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else None,
                        "sys/gpu_max_alloc_MB": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None,
                    }

                    # NEW: add separated losses for q_reg
                    if args.objective == "q_reg":
                        logdict["train/q_loss"] = float(q_loss_global)
                        logdict["train/ntp_loss"] = float(ntp_loss_global)
                        logdict["train/loss"] = logdict["train/q_loss"] + logdict["train/ntp_loss"]

                    wandb.log(logdict, step=update_step)

                update_time = time.time()

                # token-based virtual epoch boundary (optional in non-streaming)
                _maybe_epoch_hooks()

            # end of a TRUE epoch
            epoch_idx += 1
            if args.eval_each_epoch:
                total_loss, evaluated_on_tokens = evaluate_model(
                    args, model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
                )
                if global_rank == 0:
                    wandb.log({
                        "update_step": update_step,
                        "epoch": epoch_idx,
                        "eval/loss": total_loss,
                        "eval/ppl": float(np.exp(min(20, total_loss))),
                        "eval/tokens": evaluated_on_tokens,
                    }, step=update_step)
            if args.save_each_epoch and global_rank == 0:
                logger.info(f"Saving checkpoint at end of epoch {epoch_idx}")
                save_model_and_tokenizer_checkpoint(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    run_config=run_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    update_step=update_step,
                    global_step=global_step,
                    tokens_seen=tokens_seen,
                    tokens_seen_before=tokens_seen_before,
                    update_time=update_time,
                    layer_wise_flag=layer_wise_flag,
                    optimizer_dict=optimizer_dict,
                    wandb_run=wandb_run,
                )

            if update_step >= args.num_training_steps:
                break

    else:
        # ---- STEP-BOUND LOOP ----
        # For streaming datasets, re-iterate the dataloader until we hit the cap(s).
        # For non-streaming, behavior remains single pass.
        while update_step < args.num_training_steps and (args.max_train_tokens is None or tokens_seen < args.max_train_tokens):
            for batch_idx, batch in enumerate(dataloader):
                # === existing body starts ===
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

                if args.max_train_tokens is not None and tokens_seen >= args.max_train_tokens:
                    logger.info(f"Reached max_train_tokens={args.max_train_tokens}. Stopping training.")
                    break

                # --- keep the rest of your existing inner loop body unchanged ---
                loss, loss_parts = compute_loss(args, model, batch, labels)
                valid_tokens = (labels != -100).sum()
                if 'sum_loss_tok' not in locals():
                    sum_loss_tok = 0.0
                    sum_tokens_tok = 0.0
                sum_loss_tok += float(loss.detach()) * float(valid_tokens.item())
                sum_tokens_tok += float(valid_tokens.item())

                if args.objective == "q_reg":
                    if 'sum_q_loss_tok' not in locals():
                        sum_q_loss_tok = 0.0
                        sum_ntp_loss_tok = 0.0
                    if "q_loss" in loss_parts and loss_parts["q_loss"] is not None:
                        sum_q_loss_tok += float(loss_parts["q_loss"].detach()) * float(valid_tokens.item())
                    if "ntp_loss" in loss_parts and loss_parts["ntp_loss"] is not None:
                        sum_ntp_loss_tok += float(loss_parts["ntp_loss"].detach()) * float(valid_tokens.item())

                scaled_loss = loss / args.gradient_accumulation
                scaled_loss.backward()

                if global_step % args.gradient_accumulation != 0:
                    continue

                update_loss_global = ddp_token_weighted_avg(
                    sum_loss_tok, sum_tokens_tok, device=device, single_gpu=args.single_gpu
                )
                if args.objective == "q_reg":
                    q_loss_global = ddp_token_weighted_avg(sum_q_loss_tok, sum_tokens_tok, device=device, single_gpu=args.single_gpu)
                    ntp_loss_global = ddp_token_weighted_avg(sum_ntp_loss_tok, sum_tokens_tok, device=device, single_gpu=args.single_gpu)

                sum_loss_tok = 0.0
                sum_tokens_tok = 0.0
                if args.objective == "q_reg":
                    sum_q_loss_tok = 0.0
                    sum_ntp_loss_tok = 0.0

                if args.grad_clipping != 0.0:
                    try:
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping))
                    except Exception:
                        grad_norm = None
                else:
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

                if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
                    logger.info(f"Saving checkpoint at update step {update_step}")
                    save_model_and_tokenizer_checkpoint(
                        args=args,
                        model=model,
                        tokenizer=tokenizer,
                        run_config=run_config,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        update_step=update_step,
                        global_step=global_step,
                        tokens_seen=tokens_seen,
                        tokens_seen_before=tokens_seen_before,
                        update_time=update_time,
                        layer_wise_flag=layer_wise_flag,
                        optimizer_dict=optimizer_dict,
                        wandb_run=wandb_run,
                    )

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
                        }, step=update_step)
                    logger.info(f"Eval loss at step {update_step}: {total_loss}")

                lr = (optimizer.param_groups[0]["lr"]
                      if not layer_wise_flag else list(optimizer_dict.values())[0].param_groups[0]["lr"])
                tokens_in_update = tokens_seen - tokens_seen_before
                tokens_seen_before = tokens_seen
                batches_in_update = args.gradient_accumulation * world_size

                if global_rank == 0 and update_step % LOG_EVERY == 0:
                    throughput_examples = (
                        args.total_batch_size if args.total_batch_size is not None
                        else args.batch_size * args.gradient_accumulation * world_size
                    ) / max(1e-9, update_time)
                    logdict = {
                        "update_step": update_step,
                        "train/loss": float(update_loss_global),
                        "train/ppl": float(np.exp(min(20, update_loss_global))),
                        "train/lr": lr,
                        "train/tokens_seen": tokens_seen,
                        "train/throughput_tokens_s": tokens_in_update / max(1e-9, update_time),
                        "train/throughput_examples_s": throughput_examples,
                        "train/throughput_batches_s": batches_in_update / max(1e-9, update_time),
                        "train/grad_norm": grad_norm,
                        "sys/gpu_mem_alloc_MB": torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None,
                        "sys/gpu_mem_reserved_MB": torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else None,
                        "sys/gpu_max_alloc_MB": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None,
                    }
                    if args.objective == "q_reg":
                        logdict["train/q_loss"] = float(q_loss_global)
                        logdict["train/ntp_loss"] = float(ntp_loss_global)
                        # keep "train/loss" as you already had (optional):
                        logdict["train/loss"] = logdict["train/q_loss"] + logdict["train/ntp_loss"]
                    wandb.log(logdict, step=update_step)

                update_time = time.time()
                _maybe_epoch_hooks()
                # === existing inner body ends ===

            # If not streaming, do only one pass (original behavior)
            if not args.streaming:
                break

            # For streaming: we finished one pass but still haven't hit caps.
            # Just loop again; DataLoader will create a fresh iterator next pass.
            update_time = time.time()  # avoid huge elapsed time across re-iterations
    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    if global_rank == 0:
        # Save final checkpoint if not already saved for this step
        save_model_and_tokenizer_checkpoint(
            args=args,
            model=model,
            tokenizer=tokenizer,
            run_config=run_config,
            optimizer=optimizer,
            scheduler=scheduler,
            update_step=update_step,
            global_step=global_step,
            tokens_seen=tokens_seen,
            tokens_seen_before=tokens_seen_before,
            update_time=update_time,
            layer_wise_flag=layer_wise_flag,
            optimizer_dict=optimizer_dict,
            wandb_run=wandb_run,
            skip_if_exists=True,
        )

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    try:
        del optimizer, scheduler
    except Exception:
        pass
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