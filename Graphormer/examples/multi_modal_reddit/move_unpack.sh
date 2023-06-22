module load python/3.10
cp processed_graphs.tar.gz $SLURM_TMPDIR/.
cd $SLURM_TMPDIR 
tar -xvf processed_graphs.tar.gz 
cd /home/lhebert/scratch/multi-modal-reddit/code/Graphormer/examples/multi_modal_reddit 
bash reddit.sh