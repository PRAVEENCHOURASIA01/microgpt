# microgpt

A tiny GPT language model written from scratch in PyTorch, small enough to read in one sitting
(about 350 lines in total, comments included; the model itself is under 150). It learns to write text one character at a time.

```
microgpt/
  model.py        the whole GPT: attention, MLP, transformer block, text generation
  train.py        data, batches, training loop, evaluation, saving the best checkpoint
  sample.py       load a trained checkpoint and write text
  test_model.py   two tests: the model cannot see the future, and the starting loss is right
  requirements.txt
```

## Prerequisites

- Python 3.10 or newer
- PyTorch: `python3 -m pip install torch`

Nothing else is needed. On a Mac with Apple silicon it uses the GPU automatically (MPS).

## Run it

```bash
python3 test_model.py        # 1. check the model behaves (a few seconds)
python3 train.py             # 2. downloads Tiny Shakespeare (1 MB) and trains
python3 sample.py            # 3. write text with the trained model
python3 sample.py --prompt "ROMEO:" --tokens 400 --temperature 0.7
```

Train on your own text instead: `python3 train.py --data mytext.txt`

Useful settings (all optional): `--steps 3000 --batch_size 32 --block_size 128 --n_layer 4 --n_head 4 --n_embd 256 --lr 0.001`

## How it works

1. **Tokens.** Every distinct character gets an integer id (about 65 for Shakespeare). That is the tokenizer.
2. **Training examples.** Take a random window of 128 characters (`x`) and the same window shifted by one (`y`).
   At every position the correct answer is the next character.
3. **Embeddings.** A lookup table turns each id into a vector of 256 numbers. A second table adds the position.
4. **Transformer blocks (4 of them).** Each has two parts, both wrapped in a residual connection:
   - *Attention*: each token looks at the tokens before it (never after, because of the mask) and mixes in what matters.
   - *MLP*: each token processes what it gathered, on its own.
5. **Prediction.** A final layer turns each vector into one score per character. Softmax turns scores into probabilities.
6. **Learning.** Cross-entropy loss measures how little probability the correct character got.
   Backpropagation finds how to change each weight, and AdamW changes it. Repeat thousands of times.
7. **Writing.** Predict the next character, pick one at random (weighted by probability), append it, repeat.

## Measured facts

| Item | Value | How it was measured |
|---|---|---|
| Parameters (default settings, Shakespeare vocabulary of 65) | 3,208,960 | `GPT(Config()).num_params()` |
| Loss of a model guessing randomly | 4.174 = ln(65) | printed by `train.py` |
| Tests | 2 passing | `python3 test_model.py` |

Add your own results here after you train (copy them from your terminal):

| Item | Value |
|---|---|
| Device | |
| Training time | |
| Final validation loss | |
| Best validation loss | |

## Limits

A 3M-parameter character-level model trained on 1 MB of text. It learns spelling, word shapes, line structure and
the style of the text, but it does not understand meaning, so its sentences are mostly nonsense.