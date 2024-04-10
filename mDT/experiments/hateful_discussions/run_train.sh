#!/bin/bash
# Launch SLURM parameters
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --account=rcohen_group
#SBATCH --partition=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=JOB-%j.log
#SBATCH -e JOB-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y296guo@uwaterloo.ca

export SLURM_TMPDIR=`pwd`
export src=`pwd`

export WANDB_NAME=transfer-learning--$(date +%D)--$(hostname)--${RANDOM}

if [ ! -f images.tar.gz ]
then
    echo "downloading files!" 
    wget https://vault.cs.uwaterloo.ca/s/d2o9sWGrSjr7Pyd/download/duped-images.parquet
    wget https://vault.cs.uwaterloo.ca/s/EdSCwJwJ4z5cHLX/download/raw_graphs.json
    wget https://vault.cs.uwaterloo.ca/s/yKYw39ZnHZpGwNQ/download/image_indexes.tar.gz
    tar -xvf image_indexes.tar.gz 
    wget https://vault.cs.uwaterloo.ca/s/WXMwEMZpmJE7JkR/download/images.tar.gz 
fi

cp images.tar.gz $SLURM_TMPDIR/.
cp raw_graphs.json $SLURM_TMPDIR/raw_graphs.json
cp image_subset_indexes/train_index-$6-images.txt $SLURM_TMPDIR/train-idx.txt
cp image_subset_indexes/test_index-$6-images.txt $SLURM_TMPDIR/test-idx.txt
cp duped-images.parquet $SLURM_TMPDIR/duped.parquet

cd $SLURM_TMPDIR 
echo "unpacking images..."
tar -xf images.tar.gz 
echo "done unpacking!"
mkdir -p processed_graphs/processed
cd $src 

fairseq-train \
--user-dir ../../src \
--user-data-dir ./datasets \
--num-workers 8 \
--dataset-name hateful_discussions \
--task node_prediction \
--criterion node_cross_entropy \
--arch multi_graphormer_base \
--num-classes 1 \
--attention-dropout 0.3 --act-dropout 0.3 --dropout 0.4 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --weight-decay 0.01 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 500 --total-num-update 3350 \
--lr 3e-5 --end-learning-rate 3e-7 \
--spatial-pos-max $3 \
--validate-interval-updates 500 \
--batch-size 4 \
--fp16 \
--update-freq 12 \
--data-buffer-size 24 \
--encoder-layers 4 \
--num_bottleneck_tokens $2 \
--num_fusion_layers $1 \
--num_graph_stack $4 \
--num_fusion_stack $5 \
--encoder-embed-dim 768 \
--distributed-world-size 1 \
--encoder-ffn-embed-dim 768 \
--encoder-attention-heads 12 \
--max-epoch 57 \
--wandb-project "Multi-Modal Discussion Transformer" \
--save-dir "./checkpoints-final/`date +%D`/${WANDB_NAME}" \
--restore-file "/u1/y296guo/MultiModalDiscussionTransformer/mDT/experiments/hateful_discussions/contrastive-checkpoints/contrastive_checkpoint_best.pt" \
--positive-weight 1.5 \
--negative-weight 1 \
--freeze_initial_encoders \
--split $6 \
--reset-optimizer \
--batch-size-valid 16
