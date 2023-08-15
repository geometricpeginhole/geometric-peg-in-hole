export CUDA_VISIBLE_DEVICES=0
export CACHE_PATH=cache0.npy

for DATA_DIR in data_colored/1000_ypzr data_colored/1000_zpzr data_colored/1000_yzr data_colored/1000_yzpyzr data_colored/1000_yp data_colored/1000_zp data_colored/1000_yr data_colored/1000_zr 
do
    for MODEL in resnet18 clip50
    do
        python scripts/train/train.py wandb.project=geometric-peg-in-hole-colored\
            data_dir=$DATA_DIR cache_path=$CACHE_PATH env=colored model/image_encoder=$MODEL
    done
done