#!/bin/bash
# Load miniforge3 module
module load miniforge3

#module load cuda/11.8
# Initialize conda for the bash shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate your environment
ENV_PREFIX="/mnt/PRISM/cenv"
source activate $ENV_PREFIX


# Print environment info for debugging
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_PREFIX"

echo "Running transe experiments"
# Run the training script
python transe-db100k-script.py

echo "Running conve experiments"
python conve-db100k-script.py

echo "Running complex experiments"
python complex-db100k-script.py

echo "Successfully completed...."