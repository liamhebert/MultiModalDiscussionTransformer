for i in {0..6}; do
    sbatch reddit.sh 10 4 5 2 4 $i
done