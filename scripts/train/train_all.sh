export CUDA_VISIBLE_DEVICES=0
export CACHE_PATH=cache0.npy

for IMAGE_ENCODER in resnet18 resnet50 imagenet50 clip50 r3m50 imagenet_base mae_base clip_base
do
        for DATA_DIR in data_x/1000_yp data_x/1000_zp data_x/1000_yr data_x/1000_zr data_x/1000_ypzr data_x/1000_zpzr data_x/1000_yzr data_x/1000_yzpyzr
        do
                python scripts/train/train.py\
                        wandb.project=geometric-peg-in-hole-stat3\
                        model/image_encoder=$IMAGE_ENCODER\
                        data_dir=$DATA_DIR\
                        cache_path=$CACHE_PATH
        done
done
