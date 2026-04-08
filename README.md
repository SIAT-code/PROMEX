# PROMEX: Protein Meta-learning with Mixture-of-Experts

## Overview

PROMEX improves few-shot protein property prediction by combining a Mixture-of-Experts (MoE) structure with meta-learning. The pipeline consists of three stages:

1. **Cluster** — Fine-tune ESM-2 on cluster-partitioned source task datasets to obtain task-specific expert weights.
2. **MetaDistill** — Assemble experts into a MoE teacher model (via `MixtureFFNDown`) and train with a MAML-style meta-learning objective on target tasks.
3. **Finetune** — Initialize from the meta-distilled checkpoint and fine-tune on target tasks under few-shot settings.

The base protein language model is **ESM-2** (`esm2_t33_650M_UR50D`, 650M parameters).

---

## Environment

```bash
conda create -n promex python=3.10
conda activate promex
pip install torch transformers pytorch-lightning lmdb easydict peft wandb
```

---

## Data Preparation

All datasets are stored in LMDB format. Paths are specified in the YAML config files under `dataset.train_lmdb`, `dataset.valid_lmdb`, and `dataset.test_lmdb`.

**Source tasks** (used in Stage 1): AAV, EC, GO (BP/CC/MF), HumanPPI, Fluorescence, Mutant

**Target tasks** (used in Stages 2 & 3): BetaLactamase, RemoteHomology, Stability, Thermostability

Few-shot splits: `percent1` (1%), `percent5` (5%), `percent10` (10%), `percent30` (30%)

---

## Stage 1: Cluster

Fine-tune ESM-2 on each cluster subset of a source task to produce expert weights.

```bash
cd 1_Cluster
python scripts/training.py -c config/AAV/esm2-cluster0.yaml
python scripts/training.py -c config/EC/esm2-cluster1.yaml
```

Config files are organized as `config/<Task>/esm2-cluster<N>.yaml`. Key fields:

```yaml
model:
  model_py_path: saprot/saprot_regression_model   # or classification/ppi
  kwargs:
    config_path: /path/to/esm2_t33_650M_UR50D
  save_path: weights/<Task>/cluster<N>/esm2_t33_650M_UR50D.pt

dataset:
  train_lmdb: /path/to/lmdb/<task>/cluster<N>/train
```

---

## Stage 2: MetaDistill

Load Stage 1 expert weights into a grouped MoE structure and apply meta-learning on the target task. Only the aggregator in the MoE layer is trained.

```bash
cd 2_MetaDistill
python scripts/training.py -c config/BetaLactamase/moe/esm2-percent1.yaml
python scripts/training.py -c config/RemoteHomology/moe/esm2-percent5.yaml
```

Config files are organized as `config/<Task>/moe/esm2-percent<N>.yaml`. Key fields:

```yaml
model:
  teacher_name: moe
  teacher_checkpoint: /path/to/stage1/teacher.pt
  save_path: weights/<Task>/moe/esm2_t33_650M_UR50D.pt

dataset:
  iters: 128          # meta-learning iterations per epoch
  adapt_steps: 2      # inner-loop gradient steps
  adapt_batch_size: 2
  eval_batch_size: 4

Trainer:
  max_epochs: 5
```

---

## Stage 3: Finetune

Fine-tune the meta-distilled model on the target task under standard supervised training.

```bash
cd 3_Finetune
python scripts/training.py -c config/BetaLactamase/moe/esm2-normal.yaml
python scripts/training.py -c config/RemoteHomology/moe/esm2-percent1.yaml
```

Key fields:

```yaml
model:
  from_checkpoint: ../2_MetaDistill/weights/<Task>/moe/esm2_t33_650M_UR50D.pt
  save_path: weights/<Task>/esm2/moe/esm2_t33_650M_UR50D.pt

Trainer:
  max_epochs: 50
```

---

## Logging (Optional)

Training can be tracked with W&B. Set in the config file:

```yaml
setting:
  wandb_config:
    project: <project_name>
    name: <run_name>
  os_environ:
    WANDB_API_KEY: <your_key>
Trainer:
  logger: True
```
