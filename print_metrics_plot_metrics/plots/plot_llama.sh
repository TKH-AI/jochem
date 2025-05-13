#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=combinedplots
#SBATCH --cpus-per-task=14
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G

# Load the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate distill

# Use the Python executable from the distill environment to run the script
~/miniconda3/envs/distill/bin/python plot_llama.py