for l in 0 1 2 3 4 5 6 7 8 9
do
    for d in 1d 3d 5d 7d
    do
        python prep_data_step2.py \
            --data_dir .. \
            --log_num $l --data_type $d
    done
done