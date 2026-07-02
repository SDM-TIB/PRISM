#!/bin/bash
# Script to create environment with CUDA 11.8 support for GPUs
# Load miniforge
module load miniforge3
# Load Cuda - check version with (module avail cuda)
module load cuda/11.8
# Initialize conda for bash
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create a minimal environment with Python first
ENV_PREFIX="/mnt/PRISM/cenv"
echo "Creating minimal environment with Python 3.10.15..."
conda create -p $ENV_PREFIX python=3.10.15 pip -y

# Configure conda to use a shorter name for this environment
mkdir -p ~/.conda/environments.d
echo "/mnt/SPLAIN/cenv splain" > ~/.conda/environments.d/env_vars.txt
# Update .condarc to show shortened env names
if ! grep -q "env_prompt" ~/.condarc 2>/dev/null; then
  echo "env_prompt: '({name}) '" >> ~/.condarc
fi

# Activate the environment
source activate $ENV_PREFIX

# Install mamba into the environment
echo "Installing mamba into the environment..."
conda install -y mamba -c conda-forge

# Add necessary channels
conda config --env --add channels nvidia/label/cuda-11.8.0
conda config --env --add channels pytorch
conda config --env --add channels conda-forge

# Install PyTorch with CUDA support and other conda packages using mamba
echo "Installing PyTorch and CUDA packages with mamba..."
mamba install -y pytorch=2.5.0 torchvision=0.20.0 torchaudio=2.5.0 cuda=11.8.0 pytorch-cuda=11.8

echo "Installing compiler and system packages with mamba..."
mamba install -y compilers=1.5.2 sysroot_linux-64=2.17 gcc=11.4.0 py-cpuinfo=9.0.0 libaio=0.3.113

# If mamba fails on any package group, try installing them individually
# with conda using the --no-deps flag as a fallback

echo "Installing pip packages with exact versions..."
pip install tqdm ipykernel rdflib pandas pandasql SQLAlchemy torch numpy duckdb
# Verify PyTorch CUDA is working
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# Print environment information
echo "Environment setup complete"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
