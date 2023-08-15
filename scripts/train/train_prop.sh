export CUDA_VISIBLE_DEVICES=0
export CACHE_PATH=cache0.npy

for MODEL in resnet18 clip50 r3m50
do
    for PROP in no_prop
    do
        for DATA_DIR in data_x/1000_yp data_x/1000_zp data_x/1000_yr data_x/1000_zr data_x/1000_ypzr data_x/1000_zpzr data_x/1000_yzr data_x/1000_yzpyzr
        do
            python scripts/train/train.py wandb.project=geometric-peg-in-hole-prop\
                model/image_encoder=$MODEL data_dir=$DATA_DIR proprioception=$PROP\
                cache_path=$CACHE_PATH
        done
    done
done