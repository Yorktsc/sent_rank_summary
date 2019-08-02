for i in {0..299}; do
    nohup python pp.py $i > data_s/$i 2>&1 &
done
