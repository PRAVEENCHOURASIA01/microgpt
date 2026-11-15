"""
Train microgpt on a text file, one character at a time.

    python3 train.py                       # downloads Tiny Shakespeare and trains
    python3 train.py --data mytext.txt     # train on any text file you have
"""
import argparse
import math
import os
import time
import urllib.request
from dataclasses import asdict

import torch

from model import GPT, Config

# ---------------------------------------------------------------- settings
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="input.txt")
parser.add_argument("--steps", type=int, default=3000)         # how many weight updates
parser.add_argument("--batch_size", type=int, default=32)      # sequences per step
parser.add_argument("--block_size", type=int, default=128)     # context length
parser.add_argument("--n_layer", type=int, default=4)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--n_embd", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-3)          # peak learning rate
parser.add_argument("--eval_every", type=int, default=250)
parser.add_argument("--out", default="microgpt.pt")
args = parser.parse_args()

torch.manual_seed(1337)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ---------------------------------------------------------------- data
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists(args.data) and args.data == "input.txt":
    print("downloading Tiny Shakespeare...")
    urllib.request.urlretrieve(URL, args.data)

text = open(args.data, encoding="utf-8").read()
chars = sorted(set(text))                                # the vocabulary: every distinct character
stoi = {c: i for i, c in enumerate(chars)}               # character -> integer id
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]                # last 10% is never trained on
print(f"{len(text):,} characters, vocabulary of {len(chars)}")


def get_batch(split):
    """Random windows of text. y is x shifted by one: at every position, the target is the next character."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - args.block_size - 1, (args.batch_size,))
    x = torch.stack([d[i : i + args.block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + args.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- model
cfg = Config(vocab_size=len(chars), block_size=args.block_size, n_layer=args.n_layer,
             n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
model = GPT(cfg).to(device)
print(f"parameters: {model.num_params():,}")
print(f"loss of a model that guesses randomly: {math.log(len(chars)):.3f}  (= ln of vocabulary size)")
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)


def get_lr(step, warmup=100):
    """Warm up linearly, then decay along a cosine curve down to 10% of the peak."""
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, args.steps - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(iters=50):
    """Average loss over several batches, for train and for validation (text the model never trains on)."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = get_batch(split)
            losses[i] = model(x, y)[1].item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------- training loop
best_val = float("inf")
start = time.time()
for step in range(args.steps + 1):
    for group in optimizer.param_groups:
        group["lr"] = get_lr(step)

    if step % args.eval_every == 0 or step == args.steps:
        losses = estimate_loss()
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {time.time() - start:.0f}s")
        if losses["val"] < best_val:                     # keep the best checkpoint
            best_val = losses["val"]
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "chars": chars}, args.out)
    if step == args.steps:
        break

    x, y = get_batch("train")
    _, loss = model(x, y)                                # forward pass: predict, measure the error
    optimizer.zero_grad(set_to_none=True)                # forget the previous step's gradients
    loss.backward()                                      # backpropagation: how should each weight change?
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # cap huge updates
    optimizer.step()                                     # change the weights a little

print(f"best validation loss: {best_val:.4f}, saved to {args.out}")

# ---------------------------------------------------------------- see what it learned
model.eval()
start_ids = torch.zeros((1, 1), dtype=torch.long, device=device)   # begin from id 0 (the first character)
sample = model.generate(start_ids, max_new_tokens=500, temperature=0.8, top_k=40)
print("\n--- sample ---")
print("".join(chars[i] for i in sample[0].tolist()))