# 🧬 PROMEX Model Card

![Backbone](https://img.shields.io/badge/backbone-ESM--2_650M-3776AB)
![Model](https://img.shields.io/badge/model-MoE_teacher_%2B_meta--distillation-6f42c1)
![Tasks](https://img.shields.io/badge/tasks-regression%20%7C%20classification%20%7C%20ranking-2f9e44)
![Use](https://img.shields.io/badge/use-research_only-orange)

## 🔎 Model Summary

PROMEX (Protein Meta-MoE Evolution and eXploration) is a protein meta-learning framework built on the ESM-2 650M protein language model backbone. It uses a pretrained mixture-of-experts (MoE) teacher, adapts that teacher on source protein tasks, then transfers the adapted knowledge to downstream tasks through meta-distillation and fine-tuning.

This model card describes the model artifacts and intended use of the code package released with the PROMEX article.

> [!NOTE]
> PROMEX is released as a research artifact with code, prepared data, and checkpoints. Full execution commands are kept in the `run.sh` files under each task directory.

## 📦 Model Artifacts

| Artifact | Path | Purpose |
| --- | --- | --- |
| ESM-2 backbone | `weights/esm2_t33_650M_UR50D/` | Base protein sequence encoder used by PROMEX models |
| Pretrained PROMEX teacher | `weights/pretrained/moe_teacher_pretrained_model.pt` | Starting MoE teacher checkpoint before task adaptation |
| Stage 1 teacher-adapt checkpoints | `2_Property/1_Teacher/weights/*-adapt/esm2_t33_650M_UR50D.pt` | Adapted teacher models for cross-task initialization |
| Stage 2 MetaDistill checkpoints | `2_Property/2_MetaDistill/weights/<task>/normal/esm2_t33_650M_UR50D.pt` | Student checkpoints distilled for downstream property tasks |
| Stage 3 fine-tuned checkpoints | `2_Property/3_Finetune/weights/<task>/normal/promex/esm2_t33_650M_UR50D.pt` | Final fine-tuned checkpoints for downstream tasks |
| Pichia teacher-adapt checkpoint | `4_Pichia_Pastoris/1_Teacher/weights/MPB-EXP-adapt/esm2_t33_650M_UR50D.pt` | Expression-adapted teacher used by Pichia MetaDistill |
| Pichia expression MetaDistill checkpoints | `4_Pichia_Pastoris/2_MetaDistill_Expression/weights/...` | Generated student checkpoints for Pichia expression rounds |
| Pichia expression fine-tuned checkpoints | `4_Pichia_Pastoris/3_Finetune_Expression/weights/...` | Generated final expression checkpoints |

The release currently includes prepared checkpoints for the main property workflows and the Pichia teacher-adapt checkpoint. Some generated Pichia MetaDistill and fine-tuning checkpoints are produced by the `run.sh` commands in their corresponding folders.

## ✅ Intended Use

PROMEX is intended for research use in data-efficient protein modeling, including:

- protein property regression: Stability, Thermostability, Beta-Lactamase;
- protein classification: Remote Homology;
- token-level protein annotation: Secondary Structure;
- few-shot mutation fitness/ranking experiments in FS-Mutant;
- auxiliary Pichia pastoris secretion-expression and mutation-candidate workflows.

The code is designed for reproducing the PROMEX article experiments, testing prepared checkpoints, and adapting the workflow to related protein engineering datasets.

## ⚠️ Out-of-Scope Use

> [!WARNING]
> PROMEX predictions should be treated as computational evidence, not experimental proof.

PROMEX checkpoints should not be used as the sole basis for clinical, medical, environmental, or production biomanufacturing decisions. Wet-lab validation and domain review are required before using model predictions to select real experimental candidates.

## 🔁 Inputs and Outputs

For property tasks, models read LMDB records with this structure:

```json
{"name": "stab_train_0", "seq": "DQSVRKLVRKLPDEGLDREKVKTYLDKLGVDREELQKFSDAIGLESSGGS", "label": -0.209999993443489}
```

For user-provided property data, see `demo_data/property_regression_demo.csv` and convert CSV files with:

```bash
python tools/csv_to_lmdb.py --input demo_data/property_regression_demo.csv --output /tmp/promex_demo_lmdb --overwrite
```

For Pichia mutation-candidate ranking, CSV files use:

```text
mutant,mutated_sequence,DMS_score
```

The main model outputs are regression values, class logits/probabilities, token-level predictions, or mutation-ranking scores depending on the selected task and script.

## 🧭 Training and Adaptation Procedure

PROMEX uses three stages:

1. Stage 1 Teacher Adaptation: the pretrained MoE teacher is adapted on Stability, Beta-Lactamase, and Secondary-Structure for the property tasks. The Pichia branch also adapts a teacher on MPB-EXP in `4_Pichia_Pastoris/1_Teacher`.
2. Stage 2 Meta-Distillation: each downstream task initializes from a teacher-adapt checkpoint trained on a different source task, then learns a task-specific student model. Pichia expression MetaDistill uses the MPB-EXP teacher-adapt checkpoint.
3. Stage 3 Fine-Tuning: the MetaDistill student checkpoint is fine-tuned on the same downstream task for the final reported model.

The default cross-task initialization is:

| Downstream task | Stage 1 teacher-adapt source |
| --- | --- |
| Stability | Beta-Lactamase |
| Thermostability | Stability |
| Beta-Lactamase | Stability |
| Remote-Homology | Secondary-Structure |
| FS-Mutant | Secondary-Structure |
| Pichia expression | MPB-EXP |

## 🗂️ Data Used

The released package includes prepared LMDB and CSV/PKL data used by the workflows:

- `1_Data/LMDB/Stability/`
- `1_Data/LMDB/Thermostability/`
- `1_Data/LMDB/Beta-Lactamase/`
- `1_Data/LMDB/Remote-Homology/`
- `1_Data/LMDB/Secondary-Structure/`
- `3_FsMutant/data/`
- `1_Data/LMDB/MPB-EXP/cls2/`
- `1_Data/LMDB/Pichia-Pastoris/cls2/`
- `4_Pichia_Pastoris/4_Fitness_Prediction/data/Pichia-Pastoris/`

For exact demo inputs and custom data conversion, see `COMPUTE.md`.

## 📊 Evaluation

The PROMEX article evaluates the method on Remote Homology, Thermostability, Stability, Beta-Lactamase, and FS-Mutant few-shot mutation tasks. The repository also includes `3_FsMutant/results-summary.csv` with FS-Mutant summary metrics.

A quick file-level artifact check is provided for reviewers in `COMPUTE.md`. It checks representative real property data, Pichia expression data, the pretrained teacher checkpoint, and a Pichia mutation CSV without training a model. Full training and prediction commands are listed in the `run.sh` file under each task directory.

## 🚧 Limitations

- Performance depends on sequence similarity, task size, label quality, and whether the downstream task is close to the adaptation tasks.
- Few-shot results can vary with the split, random seed, and training size.
- The model uses sequence-based protein language model representations and does not replace experimental validation.
- Pichia prospective prediction workflows depend on the specific prepared checkpoints and candidate files named in the configs.

## 🖥️ Compute Requirements

The tested environment is summarized in `COMPUTE.md`:

- Ubuntu 20.04.3 LTS
- Python 3.10.20
- PyTorch 2.7.0+cu128
- Transformers 4.28.0
- NVIDIA RTX A6000 48 GB

The file-level artifact check runs in less than 1 second. Full training or reproduction runs require GPU resources and substantially longer runtime.

## 📄 License and Citation

The code is released under the MIT License. See `LICENSE`.

If you use this code or checkpoints, please cite the PROMEX article.
