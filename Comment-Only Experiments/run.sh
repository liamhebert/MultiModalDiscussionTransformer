for i in {0..6}; do
    python trainval.py -d mm-reddit -e text -s $i
done

