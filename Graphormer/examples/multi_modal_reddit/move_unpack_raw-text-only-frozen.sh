#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=60G
#SBATCH --time=12:00:0
#SBATCH --account=def-lgolab
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL

module load python/3.10
cp images.tar.gz $SLURM_TMPDIR/.
cp raw_graphs_fixed-2.json $SLURM_TMPDIR/.
cd $SLURM_TMPDIR 
tar -xvf images.tar.gz 
mkdir -p processed_graphs/processed
cd /home/lhebert/scratch/multi-modal-reddit/code/Graphormer/examples/multi_modal_reddit 
bash reddit-text-only-freeze.sh $1
