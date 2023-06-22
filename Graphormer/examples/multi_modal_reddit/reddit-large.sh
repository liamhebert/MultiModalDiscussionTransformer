#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:3
#SBATCH --cpus-per-task=18
#SBATCH --mem=80G
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

fairseq-train \
--user-dir ../../graphormer \
--user-data-dir ../../graphormer/data/custom_dataset \
--num-workers 18 \
--dataset-name multi_modal_reddit \
--task graph_prediction \
--criterion node_binary_cross_entropy \
--arch multi_graphormer_slim \
--num-classes 1 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.2 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --weight-decay 0.01 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 1049 --total-num-update 7000 \
--lr 2e-5 --end-learning-rate 1e-7 \
--batch-size 32 \
--update-freq 1 \
--fp16 \
--data-buffer-size 48 \
--encoder-layers 4 \
--num_bottleneck_tokens $2 \
--num_fusion_layers $1 \
--encoder-embed-dim 768 \
--distributed-world-size 1 \
--encoder-ffn-embed-dim 768 \
--encoder-attention-heads 32 \
--max-epoch 30 \
--save-dir ./ckpts-data-fix-fusion-$1-bottleneck-$2-projections-smallerepoch-same-output-encoder-large-batch-fixed-lr-sum-loss-bce-longer-no-vit-no-gnorm-attn-fix-bottleneck-fix-layer-fix \
--tensorboard-logdir ./runs-data-fix-fusion-$1-bottleneck-$2-projections-smallerepoch-same-output-encoder-fixed-bottlenecks-large-batch-fixed-lr-sum-loss-bce-longer-no-vit-no-gnorm-attn-fix-bottleneck-fix-layer-fix
