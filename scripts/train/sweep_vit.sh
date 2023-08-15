export CUDA_VISIBLE_DEVICES=1
export CACHE_PATH=cache1.npy
export WANDB_API_KEY=c155556dd02b1bb70ad6aefac1204a337538eae2

for IMAGE_ENCODER in clip_base
do
        for DATA_DIR in data_x/1000_yp data_x/1000_zp data_x/1000_yr data_x/1000_zr data_x/1000_ypzr data_x/1000_zpzr data_x/1000_yzr data_x/1000_yzpyzr
        do
                for seed in 0 1 2
                do
                        python scripts/train/train.py\
                                wandb.project=geometric-peg-in-hole-stat\
                                model/image_encoder=$IMAGE_ENCODER\
                                data_dir=$DATA_DIR\
                                seed=$seed\
                                cache_path=$CACHE_PATH
                done
        done
done


# python scripts/train/train.py -m\
#         wandb.project=geometric-peg-in-hole-stat\
#         model/image_encoder=imagenet_base,mae_base,clip_base\
#         data_dir=data_new/1000_ypzr,data_new/1000_zpzr,data_new/1000_yzr,data_new/1000_yzpyzr\
#         seed=0,1,2\
#         cache_path=$CACHE_PATH
