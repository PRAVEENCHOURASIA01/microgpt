"""
Write text with a trained microgpt.

    python3 sample.py
    python3 sample.py --prompt "ROMEO:" --tokens 400 --temperature 0.7
"""
import argparse

import torch

from model import GPT, Config

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="microgpt.pt")
parser.add_argument("--prompt", default="\n")
parser.add_argument("--tokens", type=int, default=500)
parser.add_argument("--temperature", type=float, default=0.8)   # lower = safer, higher = more surprising
parser.add_argument("--top_k", type=int, default=40)            # only sample among the 40 likeliest characters
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(args.ckpt, map_location="cpu")
cfg = Config(**ckpt["config"])
model = GPT(cfg)
model.load_state_dict(ckpt["model"])
model.to(device).eval()

chars = ckpt["chars"]
stoi = {c: i for i, c in enumerate(chars)}
unknown = sorted({c for c in args.prompt if c not in stoi})
if unknown:
    raise SystemExit(f"These prompt characters were not in the training text: {unknown}")

idx = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long, device=device)
out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
print("".join(chars[i] for i in out[0].tolist()))