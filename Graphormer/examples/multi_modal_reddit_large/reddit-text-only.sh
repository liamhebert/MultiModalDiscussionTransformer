# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#!/usr/bin/env bash

fairseq-train \
--user-dir ../../graphormer \
--user-data-dir ../../graphormer/data/custom_dataset \
--num-workers 24 \
--dataset-name multi_modal_reddit \
--task graph_prediction \
--criterion node_cross_entropy \
--arch text_graphormer \
--num-classes 2 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 5.0 --weight-decay 0.001 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 2333 --total-num-update 15550 \
--lr 2e-5 --end-learning-rate 1e-9 \
--batch-size 3 \
--update-freq 8 \
--fp16 \
--data-buffer-size 48 \
--encoder-layers $1 \
--num_bottleneck_tokens 1 \
--num_fusion_layers 1 \
--encoder-embed-dim 768 \
--distributed-world-size 1 \
--encoder-ffn-embed-dim 768 \
--encoder-attention-heads 12 \
--max-epoch 50 \
--save-dir ./ckpts-data-fix-encoder-layers-$1-projections-text-only \
--tensorboard-logdir ./runs-data-fix-encoder-layers-$1-projections-text-only
