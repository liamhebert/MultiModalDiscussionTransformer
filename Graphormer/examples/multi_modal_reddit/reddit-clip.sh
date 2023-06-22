# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#!/usr/bin/env bash

fairseq-train \
--user-dir ../../graphormer \
--user-data-dir ../../graphormer/data/custom_dataset \
--num-workers 24 \
--dataset-name multi_modal_reddit_clip \
--task graph_prediction \
--criterion node_cross_entropy \
--arch multi_graphormer_clip_slim \
--num-classes 2 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 69937 --total-num-update 466250 \
--lr 2e-6 --end-learning-rate 1e-9 \
--batch-size 2 \
--fp16 \
--data-buffer-size 20 \
--encoder-layers 4 \
--num_fusion_layers $1 \
--encoder-embed-dim 768 \
--distributed-world-size 1 \
--encoder-ffn-embed-dim 768 \
--encoder-attention-heads 4 \
--max-epoch 150 \
--save-dir ./ckpts-fusion-$1-clip \
--tensorboard-logdir ./runs-fusion-$1-clip
