"""download CodeSearchNet Python, sample 100K, train BPE, save token ids"""

# clone this repository
git clone https://github.com/github/CodeSearchNet.git
cd CodeSearchNet/

# download data (~3.5GB) from S3; build and run the Docker container
script/setup

