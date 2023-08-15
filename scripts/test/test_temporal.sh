export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=c155556dd02b1bb70ad6aefac1204a337538eae2

python scripts/test/test.py env/variant=zr model/action_decoder=lstm model_glob=train/train-data_new/1000_zr-top_and_wrist_cams-resnet18_lstm-with_prop-6d-mse-*/*0000.pth
