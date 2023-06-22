#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=2:59:0
#SBATCH --account=def-lgolab
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL

module load python/3.10
cp images.tar.gz $SLURM_TMPDIR/.
cp raw_graphs.json $SLURM_TMPDIR/.
cd $SLURM_TMPDIR 
tar -xvf images.tar.gz 
mkdir -p processed_graphs/processed
cd /home/lhebert/scratch/multi-modal-reddit/code/Graphormer/examples/multi_modal_reddit 
bash reddit.sh $1 $2
