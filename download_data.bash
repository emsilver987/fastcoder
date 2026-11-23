# download CodeSearchNet Python, sample 100K, train BPE, save token ids
#
# #!/usr/bin/env bash
set -euo pipefail

# Step 1: download raw parquet files (Python only, ~580 MB)
if [ ! -d "./raw" ]; then
  hf download code-search-net/code_search_net \
    --repo-type dataset \
    --include "python/*.parquet" \
    --local-dir ./codesearchnet_data
  mv ./codesearchnet_data/python/ ./raw
  rm -rf ./codesearchnet_data/
  echo "data has been imported into raw"
else
  echo "data has already been imported into raw"
fi

# Step 2: tokenizer + token streams + test functions
if [ ! -f "./data/train.bin" ]; then
  python data.py \
    --train "raw/train-*.parquet" \
    --val "raw/validation-*.parquet" \
    --test "raw/test-*.parquet"
  echo "data has been preprocessed into data"
else
  echo "data has already been preprocessed into data"
fi
