"""download CodeSearchNet Python, sample 100K, train BPE, save token ids"""

hf download sentence-transformers/codesearchnet --repo-type dataset --local-dir ./codesearchnet_data

