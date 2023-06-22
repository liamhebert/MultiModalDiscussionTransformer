module load python/3.10
cd ../../examples/multi_modal_reddit
cp images.tar.gz $SLURM_TMPDIR/.
cp raw_graphs_fixed-2.json $SLURM_TMPDIR/.
cd $SLURM_TMPDIR 
tar -xf images.tar.gz 
mkdir -p processed_graphs/processed
cd /home/lhebert/scratch/multi-modal-reddit/code/Graphormer/graphormer/evaluate

python evaluate.py \
    --user-dir ../../graphormer \
    --user-data-dir ../../graphormer/data/custom_dataset \
    --save_dir ../../examples/multi_modal_reddit/ckpts-data-fix-fusion-8-bottleneck-4-projections-smallerepoch-same-output-encoder-large-batch-fixed-lr \
    --split valid \
    --task graph_prediction \
    --dataset-name multi_modal_reddit \
    --arch multi_graphormer_slim \
    --num-classes 2 \
    --batch-size 3 \
    --fp16 \
    --seed 1 \
    --data-buffer-size 48 \
    --encoder-layers 4 \
    --num_bottleneck_tokens 4 \
    --num_fusion_layers 8 \
    --encoder-embed-dim 768 \
    --distributed-world-size 1 \
    --encoder-ffn-embed-dim 768 \
    --encoder-attention-heads 12 \

