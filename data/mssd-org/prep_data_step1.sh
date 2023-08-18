for l in 0 1 2 3 4 5 6 7 8 9 
do
    for t in 1d 3d 5d 7d
    do
        python prep_data_step1.py \
            --org_path . --save_path ./processed \
            --data_type $t --log_num $l
    done
done