# download CodeSearchNet Python, sample 100K, train BPE, save token ids


if [ ! -d "./raw" ]; then
  hf download sentence-transformers/codesearchnet --repo-type dataset --local-dir ./codesearchnet_data
  mv ./codesearchnet_data/pair/ ./raw
  rm -rf ./codesearchnet_data/
  echo "data has been imported into raw"
else
  echo "data has already been imported into raw"
fi
