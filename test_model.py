"""
Two small tests that check the most important properties of the model.

    python3 test_model.py        (or: python3 -m pytest -q test_model.py)
"""
import math

import torch

from model import GPT, Config


def make_model():
    torch.manual_seed(0)
    cfg = Config(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    return GPT(cfg).eval(), cfg


def test_cannot_see_the_future():
    """Changing tokens at positions 8..15 must not change the predictions at positions 0..7."""
    model, cfg = make_model()
    a = torch.randint(0, cfg.vocab_size, (1, 16))
    b = a.clone()
    b[0, 8:] = torch.randint(0, cfg.vocab_size, (8,))
    logits_a, _ = model(a)
    logits_b, _ = model(b)
    assert torch.allclose(logits_a[:, :8], logits_b[:, :8], atol=1e-5)   # the past is unaffected
    assert not torch.allclose(logits_a[:, 8:], logits_b[:, 8:])          # the changed part differs


def test_starting_loss_is_about_ln_vocab_size():
    """An untrained model is close to guessing randomly: loss is about ln(vocab_size)."""
    model, cfg = make_model()
    x = torch.randint(0, cfg.vocab_size, (8, 16))
    y = torch.randint(0, cfg.vocab_size, (8, 16))
    _, loss = model(x, y)
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) < 0.3, (loss.item(), expected)


if __name__ == "__main__":
    test_cannot_see_the_future()
    test_starting_loss_is_about_ln_vocab_size()
    print("all tests passed")