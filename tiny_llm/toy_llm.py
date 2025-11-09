# tiny_gpt_toy_rl.py
import math, os, time, urllib.request, random, argparse
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Data utilities
# ----------------------------

# TINY_SHAKES_URL = "https://raw.githubusercontent.com/karpathy/char-RNN/master/data/tinyshakespeare/input.txt"
TINY_SHAKES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"

def try_download_tiny_shakespeare(path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKES_URL, path)
        return True
    except Exception:
        return False

# ---- NEW: generic distractor inserter ----
def insert_distractors(seq, alphabet="abc012", p=0.0, mode="none", max_run=3):
    """
    Insert random-length runs (1..max_run) of distractor characters from `alphabet`
    with probability `p` at each boundary (infix), or at the end (suffix), or both.

    - seq: original balanced paren string e.g. '(()())'
    - Returns: possibly-augmented string, same parentheses order.
    """
    if not alphabet or p <= 0.0 or mode == "none":
        return seq

    def rand_run():
        L = random.randint(1, max_run)
        return "".join(random.choice(alphabet) for _ in range(L))

    out = []
    if mode in ("infix", "both"):
        for i, ch in enumerate(seq):
            out.append(ch)
            # boundary after each token (except last) may get noise
            if i < len(seq) - 1 and random.random() < p:
                out.append(rand_run())
        seq_out = "".join(out)
    else:
        seq_out = seq

    if mode in ("suffix", "both"):
        if random.random() < p:
            seq_out = seq_out + rand_run()

    return seq_out

def make_dyck1_random(n_sequences=50000, min_len=20, max_len=200,
                      distract_p=0.0, distract_alphabet="abc012", distract_mode="none", max_run=3):
    """Exact-length Dyck-1 sequences with optional distractor characters."""
    def gen_one(L):
        if L % 2 == 1 or L < 2:
            L = max(2, L + (L % 2))
        s = []
        depth = 0
        for t in range(L):
            rem = L - t
            if depth == rem:
                s.append(')'); depth -= 1; continue
            if depth == 0:
                s.append('('); depth += 1; continue
            can_open = (depth + 1) <= (rem - 1)
            if can_open and random.random() < 0.5:
                s.append('('); depth += 1
            else:
                s.append(')'); depth -= 1
        assert depth == 0 and len(s) == L
        core = "".join(s)
        return insert_distractors(core, distract_alphabet, distract_p, distract_mode, max_run)
    return "\n".join(gen_one(random.randint(min_len, max_len)) for _ in range(n_sequences))

def make_dyck1_greedy(n_sequences=50000, min_len=20, max_len=200,
                      distract_p=0.0, distract_alphabet="abc012", distract_mode="none", max_run=3):
    """Greedy exact-length baseline with optional distractors."""
    def gen_one(L):
        if L % 2 == 1 or L < 2:
            L = max(2, L + (L % 2))
        k = L // 2
        core = "(" * k + ")" * k
        return insert_distractors(core, distract_alphabet, distract_p, distract_mode, max_run)
    seqs = []
    for _ in range(n_sequences):
        L = random.randint(min_len, max_len)
        seqs.append(gen_one(L))
    return "\n".join(seqs)

def make_copy_dataset(n_sequences=50000, min_len=5, max_len=40, alphabet="abcd"):
    seqs = []
    for _ in range(n_sequences):
        L = random.randint(min_len, max_len)
        x = "".join(random.choice(alphabet) for _ in range(L))
        seqs.append("<s>" + x + "#" + x + "</s>")
    return "\n".join(seqs)

def make_paren_code(n_sequences=50000, min_len=20, max_len=200, emit_key=False,
                    distract_p=0.0, distract_alphabet="abc012", distract_mode="none", max_run=3):
    """Paren-code with hidden key; exact-length and balanced; optional distractors."""
    def gen_one(L):
        if L % 2 == 1 or L < 2:
            L = max(2, L + (L % 2))
        key = random.choice('01')
        s = []
        depth = 0
        for t in range(L):
            rem = L - t
            if depth == rem:
                s.append(')'); depth -= 1; continue
            if depth == 0:
                s.append('('); depth += 1; continue
            even = (depth % 2 == 0)
            ch = '(' if (key == '0' and even) or (key == '1' and not even) else ')'
            if ch == '(' and depth + 1 > rem - 1:
                ch = ')'
            if ch == ')' and depth == 0:
                ch = '('
            s.append(ch)
            depth += 1 if ch == '(' else -1
        assert depth == 0 and len(s) == L
        body = "".join(s)
        noisy = insert_distractors(body, distract_alphabet, distract_p, distract_mode, max_run)
        return (key + ":" + noisy) if emit_key else noisy
    return "\n".join(gen_one(random.randint(min_len, max_len)) for _ in range(n_sequences))

# ---- Updated validator: allow distractors transparently ----
def _check_paren_string(s, allowed_noise=None):
    allowed_noise = set(allowed_noise or "")
    depth = 0
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch in allowed_noise:
            continue  # ignore distractors
        else:
            raise AssertionError(f"Unexpected char '{ch}' not in allowed noise")
        if depth < 0:
            raise AssertionError("Went negative depth")
    if depth != 0:
        raise AssertionError("Did not close to zero")

def validate_dataset(text, expect_exact_length=False, min_len=20, max_len=200, with_key=False, allowed_noise=""):
    for i, line in enumerate(text.split("\n")):
        if not line: continue
        payload = line
        if with_key:
            assert ":" in line and line[0] in "01", f"Missing or bad key prefix on line {i}"
            payload = line.split(":", 1)[1]
        if expect_exact_length:
            # length bounds apply to the *paren-only* core; be conservative here
            core = "".join(ch for ch in payload if ch in "()")
            assert min_len <= len(core) <= max_len and len(core) % 2 == 0, f"Bad paren core length on line {i}"
        _check_paren_string(payload, allowed_noise=allowed_noise)
    print("✓ validation passed")

class CharDataset(Dataset):
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9*len(self.data))
        self.train = self.data[:n]
        self.val = self.data[n:]

    def __len__(self):
        return len(self.train) - self.block_size

    def get_vocab(self): return self.stoi, self.itos

    def __getitem__(self, idx):
        chunk = self.train[idx: idx+self.block_size+1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def collate(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys

# ----------------------------
# Tiny GPT model
# ----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        q = self.query(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = self.value(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, resid_pdrop=0.0, attn_pdrop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=4, n_head=4, n_embd=256, emb_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(emb_pdrop)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok = self.tok_emb(idx)
        pos = self.pos_emb(pos)[None, :, :]
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        # return logits and final hidden (for value/Q head)
        return logits, x

# ----------------------------
# Training / Evaluation
# ----------------------------

@dataclass
class Config:
    block_size: int = 256
    batch_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    lr: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 200
    eval_tokens: int = 20000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    dropout: float = 0.0
    compile: bool = False
    objective: str = "nll"
    checkpoint_interval: int = 200
    checkpoint_dir: str = "./tiny_llm_checkpoints"
    # NEW: dataset noise knobs (mirroring CLI)
    distract_p: float = 0.0
    distract_alphabet: str = "abc012"
    distract_mode: str = "none"
    # NEW: Q-regression objective knobs
    gamma: float = 1.0
    mix_q_ce: float = 0.0  # weight on Q loss when mixing with CE: total = mix_q_ce * L_q + (1-mix_q_ce) * L_ce
    q_neg_samples: int = 0    # number of negative tokens to sample per position (0 = use full vocab)
    use_q_head: bool = False  # whether to use a separate Q head (recommended)
    ce_head_only: bool = False

def evaluate(model, dataset, device, eval_tokens=20000):
    model.eval()
    total_nll = 0.0
    total_tok = 0
    total_acc = 0
    block_size = dataset.block_size
    max_blocks = min((len(dataset.val) - 1) // (block_size + 1), eval_tokens // block_size)

    with torch.no_grad():
        for i in range(max_blocks):
            start_idx = i * (block_size + 1)
            end_idx = start_idx + block_size + 1
            if end_idx > len(dataset.val): break
            chunk = dataset.val[start_idx:end_idx]
            X = chunk[:-1].unsqueeze(0).to(device)
            Y = chunk[1:].unsqueeze(0).to(device)
            logits, _ = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='sum')
            total_nll += loss.item()
            total_tok += Y.numel()
            pred = logits.argmax(dim=-1)
            total_acc += (pred == Y).sum().item()

    ppl = math.exp(total_nll / max(1, total_tok)) if total_tok > 0 else float('inf')
    acc = total_acc / max(1, total_tok) if total_tok > 0 else 0.0
    model.train()
    return ppl, acc

def save_checkpoint(model, optimizer, iteration, cfg, dataset, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.pt")
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': cfg,
        'vocab_size': dataset.vocab_size,
        'stoi': dataset.stoi,
        'itos': dataset.itos,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def make_text(kind, n_sequences=60000, **gen_kwargs):
    if kind == "tinyshakespeare":
        data_path = "./data/tinyshakespeare.txt"
        have = try_download_tiny_shakespeare(data_path)
        if not have:
            print("Failed to download Tiny Shakespeare; falling back to dyck_greedy.")
            return make_dyck1_greedy(n_sequences=n_sequences, **gen_kwargs)
        with open(data_path, "r", encoding="utf-8") as f:
            return f.read()
    if kind == "dyck_random":
        return make_dyck1_random(n_sequences=n_sequences, **gen_kwargs)
    if kind == "dyck_greedy":
        return make_dyck1_greedy(n_sequences=n_sequences, **gen_kwargs)
    if kind == "copy":
        return make_copy_dataset(n_sequences=n_sequences)
    if kind == "paren_code":
        return make_paren_code(n_sequences=n_sequences, **gen_kwargs)
    raise ValueError(f"unknown data kind: {kind}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dyck_greedy",
                        help="tinyshakespeare|dyck_random|dyck_greedy|copy|paren_code")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--objective", type=str, default="nll", choices=["nll", "acc_pg", "q_reg"])
    parser.add_argument("--checkpoint_dir", type=str, default="./tiny_llm_checkpoints")
    # NEW: distractor CLI
    parser.add_argument("--distract_p", type=float, default=0.0,
                        help="Per-boundary prob. of inserting distractors (0 disables).")
    parser.add_argument("--distract_alphabet", type=str, default="abc012",
                        help="Characters to use as distractors.")
    parser.add_argument("--distract_mode", type=str, default="none",
                        choices=["none","infix","suffix","both"],
                        help="Where to inject distractors.")
    # Q-regression args
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount for Q/regression targets (<=1)")
    parser.add_argument("--mix_q_ce", type=float, default=0.0, help="Mix weight for Q-loss vs CE (0 -> CE only)")
    parser.add_argument("--q_neg_samples", type=int, default=0, help="Negatives to sample per position (0 = full vocab)")
    parser.add_argument("--use_q_head", action="store_true", help="Add a separate Q head (value head) instead of reusing logits")
    parser.add_argument("--ce_head_only", action="store_true",
                        help="When objective=='q_reg', make CE loss update only model.head (detach hidden).")
    args = parser.parse_args()

    cfg = Config(block_size=args.block_size, batch_size=args.batch_size, n_layer=args.n_layer,
                 n_head=args.n_head, n_embd=args.n_embd, lr=args.lr, max_iters=args.max_iters,
                 compile=args.compile, objective=args.objective, checkpoint_dir=args.checkpoint_dir,
                 distract_p=args.distract_p, distract_alphabet=args.distract_alphabet, distract_mode=args.distract_mode,
                 gamma=args.gamma, mix_q_ce=args.mix_q_ce, q_neg_samples=args.q_neg_samples, use_q_head=args.use_q_head,
                 ce_head_only=args.ce_head_only)

    torch.manual_seed(cfg.seed); random.seed(cfg.seed)

    text = make_text(
        args.data, n_sequences=60000,
        distract_p=cfg.distract_p,
        distract_alphabet=cfg.distract_alphabet,
        distract_mode=cfg.distract_mode
    )

    # Optional: validate if using paren-like data
    if args.data in ("dyck_random","dyck_greedy","paren_code"):
        validate_dataset(
            text,
            expect_exact_length=True,
            allowed_noise=cfg.distract_alphabet,
            with_key=(args.data == "paren_code")
        )

    dataset = CharDataset(text, block_size=cfg.block_size)
    stoi, itos = dataset.get_vocab()

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=collate)

    model = TinyGPT(
        vocab_size=dataset.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        emb_pdrop=cfg.dropout, resid_pdrop=cfg.dropout, attn_pdrop=cfg.dropout
    ).to(cfg.device)

    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Attach a separate Q head optionally (value head) so logits remain a proper softmax if desired
    if cfg.use_q_head:
        # simple separate linear head mapping hidden -> Vocab-sized Q predictions
        model.q_head = nn.Linear(cfg.n_embd, dataset.vocab_size, bias=True).to(cfg.device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)
    print(f"Vocab={dataset.vocab_size}, Params≈{sum(p.numel() for p in model.parameters())/1e6:.2f}M, Objective={cfg.objective}")

    # Training
    model.train()
    it = 0
    t0 = time.time()
    for epoch in range(10**9):
        for X, Y in loader:
            X = X.to(cfg.device); Y = Y.to(cfg.device)
            logits, hidden = model(X)
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            token_acc = (logits.argmax(dim=-1) == Y).float().mean()

            if cfg.objective == "nll":
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
                gather_logp = logp.gather(dim=-1, index=Y.unsqueeze(-1)).squeeze(-1)
                batch_return = gather_logp.sum(dim=1).mean()
            elif cfg.objective == "acc_pg":
                with torch.no_grad():
                    sample_ix = torch.multinomial(p.view(-1, p.size(-1)), num_samples=1).view(X.size(0), X.size(1))
                sample_logp = logp.gather(dim=-1, index=sample_ix.unsqueeze(-1)).squeeze(-1)
                r = (sample_ix == Y).float()
                b = p.gather(dim=-1, index=Y.unsqueeze(-1)).squeeze(-1)
                adv = (r - b).detach()
                loss = -(adv * sample_logp).mean()
                batch_return = r.sum(dim=1).float().mean()
            elif cfg.objective == "q_reg":
                # Build exact returns-to-go targets under the absorbing-failure MDP
                B, T = Y.size()
                with torch.no_grad():
                    V_targets = torch.zeros(B, T, device=Y.device)
                    V_next = torch.zeros(B, device=Y.device)
                    for t in reversed(range(T)):
                        V_next = 1.0 + cfg.gamma * V_next
                        V_targets[:, t] = V_next

                V_targets_exp = V_targets.unsqueeze(-1)  # (B,T,1)

                # Choose prediction head for Q: separate q_head if available, otherwise reuse logits
                if cfg.use_q_head and hasattr(model, 'q_head'):
                    q_pred = model.q_head(hidden)  # (B,T,V)
                else:
                    q_pred = logits

                vocab_size = logits.size(-1)
                if cfg.q_neg_samples and cfg.q_neg_samples > 0:
                    # Negative-sample K tokens per position (plus the GT token)
                    K = cfg.q_neg_samples
                    # indices: (B,T,1) for GT
                    gt_idx = Y.unsqueeze(-1)
                    # sample K negatives uniformly (may include GT occasionally)
                    neg_idx = torch.randint(0, vocab_size, (B, T, K), device=Y.device)
                    idxs = torch.cat([gt_idx, neg_idx], dim=-1)  # (B,T,K+1)
                    q_slice = q_pred.gather(dim=-1, index=idxs)  # (B,T,K+1)
                    y_slice = torch.zeros_like(q_slice)
                    y_slice[..., 0] = V_targets
                    loss_q = 0.5 * (q_slice - y_slice).pow(2).mean()
                else:
                    # Full-vocab regression target: GT -> V_targets, others -> 0
                    Q_targets = torch.zeros_like(logits)
                    Q_targets.scatter_(dim=-1, index=Y.unsqueeze(-1), src=V_targets_exp)
                    loss_q = 0.5 * (q_pred - Q_targets).pow(2).mean()

                # CE path: optionally block gradients to backbone so only model.head updates
                if cfg.ce_head_only:
                    # detach hidden so gradients flow only into model.head
                    logits_ce = model.head(hidden.detach())
                else:
                    # normal CE uses full logits (backbone receives grads)
                    logits_ce = logits
                loss_ce = F.cross_entropy(logits_ce.view(-1, vocab_size), Y.view(-1))
                if cfg.mix_q_ce and cfg.mix_q_ce > 0.0:
                    if cfg.ce_head_only:
                        loss = loss_q + loss_ce
                    else:
                        loss = cfg.mix_q_ce * loss_q + (1.0 - cfg.mix_q_ce) * loss_ce
                else:
                    loss = loss_q

                # --- CHANGE: keep detached copies for logging ---
                log_loss_q = loss_q.detach()
                log_loss_ce = loss_ce.detach()

                # batch_return: average V target value (informative metric)
                batch_return = V_targets.mean()
            else:
                raise ValueError(f"unknown objective: {cfg.objective}")

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            # Save checkpoint every checkpoint_interval steps (default 200)
            if it % cfg.checkpoint_interval == 0 and it > 0:
                save_checkpoint(model, optim, it, cfg, dataset, cfg.checkpoint_dir)

            if it % cfg.eval_interval == 0:
                ppl, acc = evaluate(model, dataset, cfg.device, eval_tokens=cfg.eval_tokens)
                dt = time.time()-t0; t0 = time.time()
                if cfg.objective == "q_reg":
                    extra = f" | loss_q {log_loss_q.item():.3f} | loss_ce {log_loss_ce.item():.3f}"
                else:
                    extra = ""
                print(f"iter {it:5d} | obj {cfg.objective} | loss {loss.item():.3f}{extra} | "
                      f"return {batch_return.item():.3f} | tok-acc {token_acc.item():.3f} | "
                      f"val ppl {ppl:.2f} acc {acc:.3f} | {dt:.1f}s")
            it += 1
            if it >= cfg.max_iters:
                break
        if it >= cfg.max_iters:
            break

    # Save final checkpoint
    save_checkpoint(model, optim, it, cfg, dataset, cfg.checkpoint_dir)

    # Sample a bit
    model.eval()
    if '<' in stoi and 's' in stoi and '>' in stoi:
        ctx = "<s>"
    elif '0' in stoi:
        ctx = "0"
    else:
        ctx = "("
    x = torch.tensor([[stoi.get(c, 0) for c in ctx[-cfg.block_size:]]], dtype=torch.long, device=cfg.device)
    out = [c for c in ctx]
    with torch.no_grad():
        for _ in range(200):
            logits, _ = model(x)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            ch = dataset.itos[ix.item()]
            out.append(ch)
            x = torch.cat([x, ix], dim=1)
            x = x[:, -cfg.block_size:]
    print("=== SAMPLE ===")
    print("".join(out))

if __name__ == "__main__":
    main()
