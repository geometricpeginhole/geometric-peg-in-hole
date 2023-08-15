export CUDA_VISIBLE_DEVICES=0

for SEED in 0 100 200
do
    for ENV_MODEL in "env/variant=ypzr model_glob=train/train-data/1000_ypzr-top_and_wrist_cams-resnet18_mlp-with_prop-6d-mse-2023.08.11-14.29.25/*50000.pth" "env/variant=zpzr model_glob=train/train-data/1000_zpzr-top_and_wrist_cams-resnet18_mlp-with_prop-6d-mse-2023.08.11-15.38.25/*50000.pth" "env/variant=yzr model_glob=train/train-data/1000_yzr-top_and_wrist_cams-resnet18_mlp-with_prop-6d-mse-2023.08.11-16.46.44/*50000.pth" "env/variant=yzpyzr model_glob=train/train-data/1000_yzpyzr-top_and_wrist_cams-resnet18_mlp-with_prop-6d-mse-2023.08.11-17.58.50/*50000.pth"
    do 
        for OBJECTSET in arrow circle cross diamond hexagon key line pentagon u
        do
            python scripts/test/test.py env/object_set=$OBJECTSET\
                wandb.project=geometric-peg-in-hole-each\
                eval.seed_start=$SEED\
                $ENV_MODEL
        done
    done
done
