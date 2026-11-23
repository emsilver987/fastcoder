""" onetime-preprocessing fastcoder """

import argparse
import glob
import json
import os
import random

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"
CODE_COLUMNS = ["whole_func_string", "original_string", "func_code_string", "code"]
REPO_COLUMNS = ["repository_name", "repo"]


def load_functions(pattern):
    """Load every parquet shard matching the glob. Returns (code list, repo list or None)."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern}")
    codes, repos = [], []
    for path in paths:
        names = pq.read_schema(path).names  # read column names without loading the data
        col = next((c for c in CODE_COLUMNS if c in names), None)
        if col is None:
            raise ValueError(f"{path}: none of {CODE_COLUMNS} found. Columns are: {names}")
        repo_col = next((c for c in REPO_COLUMNS if c in names), None)
        # Only load the columns we need: the shards also hold token lists, URLs, etc.
        df = pd.read_parquet(path, columns=[col] + ([repo_col] if repo_col else []))
        print(f"{os.path.basename(path)}: {len(df):,} rows, using column '{col}'")
        codes += df[col].tolist()
        repos += df[repo_col].tolist() if repo_col else [None] * len(df)
    return codes, (repos if all(r is not None for r in repos) else None)


def keep(f, max_chars):
    # Do NOT strip or normalize whitespace: indentation is Python syntax.
    # Only drop empty functions and extreme outliers (huge generated tables, etc.).
    return isinstance(f, str) and f.strip() and len(f) <= max_chars


def clean(funcs, max_chars):
    return [f for f in funcs if keep(f, max_chars)]


def split_by_repo(codes, repos, holdout_frac, rng):
    """Carve val/test out of train by holding out whole repositories.
    Functions in the same repo are often near-copies, so a random per-function
    split would leak test-like code into training."""
    unique = sorted(set(repos))
    rng.shuffle(unique)
    n_hold = max(2, int(len(unique) * holdout_frac))
    val_repos = set(unique[: n_hold // 2])
    test_repos = set(unique[n_hold // 2 : n_hold])
    out = {"train": [], "val": [], "test": []}
    for c, r in zip(codes, repos):
        out["val" if r in val_repos else "test" if r in test_repos else "train"].append(c)
    print(f"repo split: {len(unique):,} repos -> {len(val_repos):,} val, {len(test_repos):,} test")
    return out


def sample(funcs, n, rng):
    # Fixed-seed sampling -> rerunning the script gives the exact same subset.
    return funcs if n >= len(funcs) else rng.sample(funcs, n)


def train_tokenizer(funcs, vocab_size):
    # Byte-level BPE: any character falls back to raw bytes, so there is never an <unk>.
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
    # One long stream with <|endoftext|> between functions. train.py samples
    # random fixed-length windows from it: no padding, no wasted compute.
    eot_id = tok.token_to_id(EOT)
    ids = []
    for enc in tok.encode_batch(funcs):
        ids.extend(enc.ids)
        ids.append(eot_id)
    return np.array(ids, dtype=np.uint16)  # 2 bytes/token; fine since vocab < 65,536


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help='glob, e.g. "raw/train-*.parquet" (quote it)')
    p.add_argument("--val", help="glob for official validation shards (optional)")
    p.add_argument("--test", help="glob for official test shards (optional)")
    p.add_argument("--holdout_frac", type=float, default=0.10,
                   help="fraction of repos held out for val+test when --val/--test are not given")
    p.add_argument("--out", default="data")
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

    if args.val and args.test:
        pools = {name: clean(load_functions(pat)[0], args.max_chars)
                 for name, pat in [("train", args.train), ("val", args.val), ("test", args.test)]}
    elif args.val or args.test:
        raise SystemExit("give both --val and --test, or neither")
    else:
        codes, repos = load_functions(args.train)
        if repos is None:
            raise SystemExit("no repository column found, so I can't do a leak-free split; "
                             "download the official validation/test files instead")
        kept = [(c, r) for c, r in zip(codes, repos) if keep(c, args.max_chars)]
        pools = split_by_repo([c for c, _ in kept], [r for _, r in kept], args.holdout_frac, rng)

    n_wanted = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    splits = {name: sample(pools[name], n_wanted[name], rng) for name in ["train", "val", "test"]}

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
