"""
microgpt: a tiny GPT, written to be read.

Shape letters used in comments:
    B = batch size, T = sequence length (time), C = embedding size (channels),
    nh = number of heads, hs = head size (C // nh)
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 65     # number of different tokens (set from the data)
    block_size: int = 128    # context length: how many tokens the model can look at
    n_layer: int = 4         # number of transformer blocks
    n_head: int = 4          # attention heads per block
    n_embd: int = 256        # size of each token's vector
    dropout: float = 0.1     # randomly drop activations while training (less overfitting)


class Attention(nn.Module):
    """Causal self-attention: every token gathers information from the tokens before it."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # makes query, key, value in one go
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)      # mixes the heads back together
        self.drop = nn.Dropout(cfg.dropout)
        # lower-triangular matrix of ones: row t has ones for positions 0..t (the past)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):                                   # x: (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)               # each (B, T, C)
        # cut the channels into heads: (B, T, C) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # how well does each query match each key?  (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        # forbid looking at the future: set those scores to -inf so softmax gives them 0
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)                        # each row now sums to 1
        att = self.drop(att)

        y = att @ v                                         # weighted average of values (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # glue the heads back: (B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    """Feed-forward network: each token 'thinks' on its own about what attention gathered."""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention, then MLP. Each is wrapped in a residual connection."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual: add the result to the input, don't replace it
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token id -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # position -> vector
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # vector -> score per token
        self.head.weight = self.tok_emb.weight   # weight tying: input and output share one table
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):                    # idx: (B, T) integer token ids
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)            # (B, T, C): what + where
        x = self.drop(x)
        x = self.blocks(x)                                   # (B, T, C)
        x = self.ln_f(x)
        logits = self.head(x)                                # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Write text: predict the next token, sample it, append it, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]         # keep only the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature          # scores for the next token only (B, vocab)
            if top_k is not None:                            # keep only the k best candidates
                kth_best = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth_best, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # random pick, weighted by probability
            idx = torch.cat([idx, next_id], dim=1)
        return idx