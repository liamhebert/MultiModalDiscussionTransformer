
# fusion_layer num_bottleneck attn_size num_graph_stack num_fusion_stack

#sbatch reddit.sh 0 4

#sbatch reddit.sh 4 4 2
# sbatch reddit.sh 4 4 5 2 2
# sbatch reddit.sh 4 4 5 4 2
#sbatch reddit.sh 6 4
#sbatch reddit.sh 8 4 2 
#sbatch reddit.sh 8 4 5 4 4

# bottleneck experiments
sbatch reddit.sh 8 4 2 2 4 1
sbatch reddit.sh 8 4 2 2 2 2
sbatch reddit.sh 8 4 2 2 4 4
sbatch reddit.sh 4 4 2 2 2 2
#sbatch reddit.sh 8 4 1024 2 4 1

sbatch reddit.sh 8 8 5 2 4 1
sbatch reddit.sh 8 8 1024 2 4 1

# sbatch reddit.sh 8 16 5 2 4 1
# sbatch reddit.sh 8 16 1024 2 4 1

# sbatch reddit.sh 8 32 5 2 4 1
# sbatch reddit.sh 8 32 1024 2 4 1

# # fusion level
sbatch reddit.sh 6 4 5 2 4 1
sbatch reddit.sh 6 4 1024 2 4 1

# sbatch reddit.sh 8 4 5 2 4
# sbatch reddit.sh 8 4 1024 2 4

sbatch reddit.sh 10 4 5 2 4 1
sbatch reddit.sh 10 4 1024 2 4 1

# sbatch reddit.sh 12 4 5 2 4 1
# sbatch reddit.sh 12 4 1024 2 4 1
#sbatch reddit.sh 8 4 5 2 2

#sbatch reddit.sh 8 4 1024 2 2
# #sbatch reddit.sh 10 4 5
# sbatch reddit.sh 10 4 2 2 2
#sbatch reddit.sh 10 8
#sbatch move_unpack_raw.sh 10 16
#sbatch reddit.sh 10 32

# sbatch reddit-large.sh 4 4
# sbatch reddit-large.sh 6 4
# sbatch reddit-large.sh 8 4
# sbatch reddit-large.sh 10 4
# sbatch reddit-large.sh 10 8
# #sbatch move_unpack_raw.sh 10 16
# sbatch reddit-large.sh 10 32