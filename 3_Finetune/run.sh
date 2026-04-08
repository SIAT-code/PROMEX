CUDA_VISIBLE_DEVICES=0 python scripts/training.py -c config/RemoteHomology/esm2-normal.yaml

CUDA_VISIBLE_DEVICES=1 python scripts/training.py -c config/Thermostability/random/supervised/esm2-percent30.yaml

CUDA_VISIBLE_DEVICES=1 python scripts/training.py -c config/BetaLactamase/random/moe3/esm2-percent1.yaml