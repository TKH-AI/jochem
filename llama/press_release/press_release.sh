#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=4presdistill
#SBATCH --cpus-per-task=14
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G

# Load the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate distill

# Use the Python executable from the distill environment to run the script
~/miniconda3/envs/distill/bin/python press_release.py
