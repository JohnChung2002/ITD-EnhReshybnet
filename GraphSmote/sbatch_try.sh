#!/bin/bash
#SBATCH --job-name=GraphSmote_Recon
#SBATCH --output=GraphSmote_Recon_%j.out
#SBATCH --time=8:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=1

module load mamba
mamba activate initial_py
cd /fred/oz382/dataset/CERT/r5.2/EnhancedResHybnet/GraphSmote
python modified_main.py --dataset cert --data_path /fred/oz382/dataset/CERT/r5.2/rs_data --setting recon --version r5.2