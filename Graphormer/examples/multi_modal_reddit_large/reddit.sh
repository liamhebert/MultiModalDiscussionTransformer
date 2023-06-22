#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=2:59:0
#SBATCH --account=def-lgolab
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL

module load python/3.10 StdEnv/2020  gcc/9.3.0  cuda/11.0 arrow/10.0.1 
cp images.tar.gz $SLURM_TMPDIR/.
cp pruned-with-images.json $SLURM_TMPDIR/.
cp train_index-0.txt $SLURM_TMPDIR/train-idx.txt
cp test_index-0.txt $SLURM_TMPDIR/test-idx.txt
cd $SLURM_TMPDIR 
echo "unpacking images..."
tar -xf images.tar.gz 
echo "done unpacking!"
mkdir -p processed_graphs/processed
cd /home/lhebert/scratch/multi-modal-reddit/code/Graphormer/examples/multi_modal_reddit_large 

fairseq-train \
--user-dir ../../graphormer \
--user-data-dir ../../graphormer/data/custom_dataset \
--num-workers 12 \
--dataset-name multi_modal_reddit \
--task graph_prediction \
--criterion node_cross_entropy \
--arch multi_graphormer_slim \
--num-classes 1 \
--attention-dropout 0.3 --act-dropout 0.3 --dropout 0.4 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --weight-decay 0.01 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 1200 --total-num-update 8250 \
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
--save-dir "./checkpoints/`date +%D`/`hostname`-${RANDOM}" \
--positive-weight 1.5 \
--negative-weight 1 

#--tensorboard-logdir ./runs-data-fix-fusion-$1-bottleneck-$2-projections-$3-position-$4-graph_stack-$5-fusion_stack-nofreeze-weighted-img-2-node-head12-b-sub2-dist-fix-drop

# --save-dir ./ckpts-data-fix-fusion-$1-bottleneck-$2-projections-$3-position-$4-graph_stack-$5-fusion_stack-freeze-weighted-img-2-node-head12-b-sub2-dist-fix-split-$6 \
# --tensorboard-logdir ./runs-data-fix-fusion-$1-bottleneck-$2-projections-$3-position-$4-graph_stack-$5-fusion_stack-freeze-weighted-img-2-node-head12-b-sub2-dist-fix-split-$6
