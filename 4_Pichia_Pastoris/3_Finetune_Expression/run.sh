CUDA_VISIBLE_DEVICES=2 python scripts/training.py -c config/promex/promex-binary-cv-round1.yaml

CUDA_VISIBLE_DEVICES=2 python scripts/training.py -c config/supervised/esm2-binary-cv-round1.yaml

CUDA_VISIBLE_DEVICES=2 python scripts/training.py -c config/promex/promex-binary-cv-round2.yaml

CUDA_VISIBLE_DEVICES=2 python scripts/training.py -c config/supervised/esm2-binary-cv-round2.yaml

CUDA_VISIBLE_DEVICES=2 python scripts/training.py -c config/promex/promex-binary-cv-round3.yaml

CUDA_VISIBLE_DEVICES=2 python scripts/training.py -c config/supervised/esm2-binary-cv-round3.yaml
