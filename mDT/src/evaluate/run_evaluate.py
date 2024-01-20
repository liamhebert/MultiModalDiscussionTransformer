--user-dir ../../src --user-data-dir ../../src/data/custom_dataset --num-workers 8 --seed 23 --dataset-name multi_modal_reddit --task graph_prediction --criterion node_cross_entropy --arch multi_graphormer_base --num-classes 1 --attention-dropout 0.3 --act-dropout 0.3 --dropout 0.4 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --weight-decay 0.01 --lr-scheduler polynomial_decay --power 2 --warmup-updates 13392 --total-num-update 44640 --lr 3e-5 --spatial-pos-max 5 --batch-size 110 --batch-size-valid 96 --fp16 --fp16-init-scale 4 --data-buffer-size 10 --encoder-layers 4 --num_bottleneck_tokens 4 --num_fusion_layers 8 --num_graph_stack 2 --num_fusion_stack 4 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 12 --max-epoch 30 --wandb-project "Multi-Modal Community Transformer" --save-dir ./checkpoints-final/12/01/23/watgpu208-19863 --positive-weight 1.5 --negative-weight 1 --freeze_initial_encoders --find-unused-parameters