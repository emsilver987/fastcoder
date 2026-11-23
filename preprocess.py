""" preprocess fastcoder """

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from tokenizeres import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"

# columns given in the dataset
CODE_COLUMNS = ["whole_func_string", "original_string", "func_code_string", "code"]

def load_functions(path):
    df = pd.read_parquet(path)
    col = next((c for c in CODE_COLUMNS if c in df.columns), None)
    if col is None:
        raise ValueError(f"{path}: none of {CODE_COLUMNS} found. Columns are: {list(df.columns)}")
        print(f"{os.path.basename(path)}: {len(df):,} rows, using column '{col}'")

    return df[col].tolist()

def clean(funcs, max_chars):
    """ drop empty funcs and huge tables """
    return [f for f in funcs if isinstance(f, str) and f.strip() and len(f) <= max_chars]

def sample(funcs, n, rng):
    """ fixed seed sampling """
    return funcs if n >= len(funs) else rng.sample(funcs, n)

def train_tokenizers(funcs, vocab_size):
    """ byte-level BPE """
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[EOT],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(funcs, trainer=trainer)
    return tok

def to_stream(tok, funcs):
    """ long stream with EOT between funcs, no padding """
    eot_id = tok.token_to_id(EOT)
    ids = []
    for enc in tok.encode_batch(funcs):
        ids.extend(enc.ids)
        ids.append(eot_id)
    return np.array(ids, dtype=np.uint16)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--n_train", type=int, default=100_000)
    p.add_argument("--n_val", type=int, default=5_000)
    p.add_argument("--n_test", type=int, default=2_000)
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--max_chars", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    assert args.vocab_size < 2**16, "uint16 storage needs vocab < 65,536"
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    splits = {}
    for name, path, n in [("train", args.train, args.n_train),
                          ("val", args.val, args.n_val),
                          ("test", args.test, args.n_test)]:
        funcs = clean(load_functions(path), args.max_chars)
        splits[name] = sample(funcs, n, rng)

    # Tokenizer sees the training subset only, so val/test stay fully unseen.
    tok = train_tokenizer(splits["train"], args.vocab_size)
    tok.save(os.path.join(args.out, "tokenizer.json"))

    for name in ["train", "val"]:
        arr = to_stream(tok, splits[name])
        arr.tofile(os.path.join(args.out, f"{name}.bin"))
        n_funcs = len(splits[name])
        print(f"{name}: {n_funcs:,} functions, {len(arr):,} tokens, "
              f"{len(arr) / n_funcs:.0f} tokens/function avg")

    # Test functions stay whole: evaluate.py needs function boundaries to cut a
    # prefix and compare the generated next line against the real one.
    with open(os.path.join(args.out, "test_functions.jsonl"), "w") as f:
        for code in splits["test"]:
            f.write(json.dumps({"code": code}) + "\n")
    print(f"test: {len(splits['test']):,} functions saved whole")

    # Sanity check: decoding must give back the exact original text, indentation included.
    sample_fn = splits["train"][0]
    assert tok.decode(tok.encode(sample_fn).ids) == sample_fn, "tokenizer round-trip failed"
    print(f"vocab size: {tok.get_vocab_size():,}  |  round-trip check passed")


if __name__ == "__main__":
    main()


