#!/bin/bash

export SLURM_TMPDIR=<INSERT_DATA_DIR>
export src=`pwd`

if [! -f raw_graphs_image_subset.json]
then
    echo "downloading files!" 
    wget https://vault.cs.uwaterloo.ca/s/AMQjNprPxsi4TBr/download/duped-images.parquet
    wget https://vault.cs.uwaterloo.ca/s/ree4cN7ajkoZXQF/download/raw_graphs_image_subset.json 
    wget https://vault.cs.uwaterloo.ca/s/g5KmEicpqEpMrTp/download/image_subset_indexes.tar.gz 
    tar -xvf image_subset_indexes.tar.gz 
    wget https://vault.cs.uwaterloo.ca/s/TfMam3Z2Q93BsML/download/images_bigger.tar.gz 
fi

cp images.tar.gz $SLURM_TMPDIR/.
cp raw_graphs_image_subset.json $SLURM_TMPDIR/raw_graphs.json
cp image_subset_indexes/train_index-$6.txt $SLURM_TMPDIR/train-idx.txt
cp image_subset_indexes/test_index-$6.txt $SLURM_TMPDIR/test-idx.txt
cp duped-images.parquet $SLURM_TMPDIR/duped.parquet

cd $SLURM_TMPDIR 
echo "unpacking images..."
tar -xf images.tar.gz 
echo "done unpacking!"
mkdir -p processed_graphs/processed
cd $src 

fairseq-train \
--user-dir ../../src \
--user-data-dir ../../src/data/custom_dataset \
--num-workers 12 \
--dataset-name multi_modal_reddit \
--task graph_prediction \
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
--max-epoch 10 \
--wandb-project "Multi-Modal Discussion Transformer" \
--save-dir "./checkpoints-final/`date +%D`/`hostname`-${RANDOM}" \
--positive-weight 1.5 \
--negative-weight 1 \
--freeze_initial_encoders \
--split $6 
