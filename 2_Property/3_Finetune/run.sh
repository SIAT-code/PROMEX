
CUDA_VISIBLE_DEVICES=2 python scripts/training.py -c config/Remote-Homology/promex-normal.yaml

CUDA_VISIBLE_DEVICES=3 python scripts/training.py -c config/Thermostability/promex-normal.yaml

CUDA_VISIBLE_DEVICES=4 python scripts/training.py -c config/Stability/promex-normal.yaml

CUDA_VISIBLE_DEVICES=2 python scripts/training.py -c config/Beta-Lactamase/promex-normal.yaml






