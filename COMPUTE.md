# ✅ PROMEX Compute and Reproducibility

![Environment](https://img.shields.io/badge/environment-fixed-blue)
![GPU](https://img.shields.io/badge/GPU-RTX_A6000_48GB-76B900?logo=nvidia&logoColor=white)
![Demo](https://img.shields.io/badge/demo-files_checked-success)
![Checklist](https://img.shields.io/badge/reviewer_checklist-complete-6f42c1)

This document summarizes the practical information needed by editors and reviewers to check whether the PROMEX code package can be installed and run. All required items are checked; only full reproduction of all experiments is optional because it is computationally expensive.

> [!IMPORTANT]
> The goal of this file is practical reproducibility: environment, hardware, demo data, commands, expected output, runtime, data format, and license.

## 📌 Required Content Checklist

| Item | Status | Location / command |
| --- | --- | --- |
| `requirements.txt` or `environment.yml` | ✓ | `environment.yml` |
| Tested environment and GPU | ✓ | Ubuntu 20.04.3 LTS; Python 3.10.20; PyTorch 2.7.0+cu128; CUDA runtime 12.8; Transformers 4.28.0; NVIDIA RTX A6000 48 GB |
| Installation time | ✓ | `conda env create -f environment.yml`; dry-run solve was verified on linux-64; budget about 20-40 minutes for a full install, depending on network/package cache |
| Small demo data | ✓ | Bundled real data are available under `1_Data/LMDB/`, `3_FsMutant/data/`, and `4_Pichia_Pastoris/4_Fitness_Prediction/data/`. `demo_data/property_regression_demo.csv` contains six real property-regression rows for CSV-to-LMDB conversion. |
| One-command demo | ✓ | See the artifact-check command in `Small Demo` below |
| Expected output | ✓ | See `Expected demo output` below |
| Demo runtime | ✓ | Less than 1 second on the tested server; no model training is performed |
| User data input format and conversion method | ✓ | Property CSV with `name,seq,label` converted by `tools/csv_to_lmdb.py`; Pichia mutation CSV with `mutant,mutated_sequence,DMS_score` preprocessed by `4_Pichia_Pastoris/4_Fitness_Prediction/preprocess.py` |
| LICENSE | ✓ | `LICENSE` |
| Full reproduction of all experiments | Optional | Run the `run.sh` files under `2_Property/1_Teacher`, `2_Property/2_MetaDistill`, `2_Property/3_Finetune`, `3_FsMutant`, and each `4_Pichia_Pastoris` subfolder |

## 🖥️ Tested Environment

| Component | Tested value |
| --- | --- |
| OS | Ubuntu 20.04.3 LTS, Linux 5.4.0-216-generic |
| GPU | NVIDIA RTX A6000, 49140 MiB memory |
| Driver | 580.126.09 |
| Python | 3.10.20 |
| PyTorch | 2.7.0+cu128 |
| CUDA runtime | 12.8 |
| Transformers | 4.28.0 |

Full package snapshot:

```text
OS: Ubuntu 20.04.3 LTS, Linux 5.4.0-216-generic
GPU: NVIDIA RTX A6000, 49140 MiB memory, driver 580.126.09
Server: 4 x RTX A6000 available; example commands use one GPU through CUDA_VISIBLE_DEVICES
Python: 3.10.20
PyTorch: 2.7.0+cu128
CUDA runtime reported by PyTorch: 12.8
Transformers: 4.28.0
PyTorch Lightning: 1.8.3
TorchMetrics: 0.9.3
LMDB: 1.5.1
Pandas: 2.2.3
scikit-learn: 1.4.2
NumPy: 1.26.4
```

## ⚙️ Installation

Create the environment from the project root:

```bash
conda env create -f environment.yml
conda activate promex
```

The `environment.yml` file was generated from `/data1/zhen/RA/Projects/Meta-MOE/PROMEX/venv` and checked with `conda env create --dry-run` on linux-64.

Expected installation time is about 20-40 minutes on a typical Linux workstation/server. The main variable is network speed and whether PyTorch/CUDA wheels are already cached.

## 🧪 Small Demo

Run this command from the project root:

```bash
bash -lc '
set -e
echo "[1/6] Checking 2_Property LMDB data"
test -d 1_Data/LMDB/Stability/normal/train
echo "      found: 1_Data/LMDB/Stability/normal/train"

echo "[2/6] Checking FS-Mutant structure table"
test -f 3_FsMutant/data/struc_sq_72.csv
echo "      found: 3_FsMutant/data/struc_sq_72.csv"

echo "[3/6] Checking FS-Mutant preprocessed task bundle"
test -f 3_FsMutant/data/merged.pkl
echo "      found: 3_FsMutant/data/merged.pkl"

echo "[4/6] Checking Pichia expression LMDB data"
test -d 1_Data/LMDB/Pichia-Pastoris/cls2/train_cv3/fold3/train
echo "      found: 1_Data/LMDB/Pichia-Pastoris/cls2/train_cv3/fold3/train"

echo "[5/6] Checking pretrained PROMEX teacher checkpoint"
test -f weights/pretrained/moe_teacher_pretrained_model.pt
echo "      found: weights/pretrained/moe_teacher_pretrained_model.pt"

echo "[6/6] Checking Pichia mutation CSV"
test -f 4_Pichia_Pastoris/4_Fitness_Prediction/data/Pichia-Pastoris/substitutions_round1/single_mutant/DeltaE/Pel114_pectate_lyase.csv
echo "      found: 4_Pichia_Pastoris/4_Fitness_Prediction/data/Pichia-Pastoris/substitutions_round1/single_mutant/DeltaE/Pel114_pectate_lyase.csv"

echo "PROMEX demo files OK"
'
```

This one-command check verifies that representative real property data, real FS-Mutant files, Pichia expression data, the pretrained teacher checkpoint, and a Pichia mutation CSV are present. It does not train a model.

> [!TIP]
> This demo is intentionally lightweight. It proves that the release contains the expected real data and checkpoint artifacts before reviewers start GPU training runs.

For training or prediction runs, use the `run.sh` file in each task directory as the canonical command list:

| Workflow | Command file |
| --- | --- |
| Property teacher adaptation | `2_Property/1_Teacher/run.sh` |
| Property MetaDistill | `2_Property/2_MetaDistill/run.sh` |
| Property fine-tuning | `2_Property/3_Finetune/run.sh` |
| FS-Mutant benchmark | `3_FsMutant/run.sh` |
| Pichia teacher adaptation | `4_Pichia_Pastoris/1_Teacher/run.sh` |
| Pichia expression MetaDistill | `4_Pichia_Pastoris/2_MetaDistill_Expression/run.sh` |
| Pichia expression fine-tuning | `4_Pichia_Pastoris/3_Finetune_Expression/run.sh` |
| Pichia fitness/ranking prediction | `4_Pichia_Pastoris/4_Fitness_Prediction/run.sh` |

Expected demo output:

```text
[1/6] Checking 2_Property LMDB data
      found: 1_Data/LMDB/Stability/normal/train
[2/6] Checking FS-Mutant structure table
      found: 3_FsMutant/data/struc_sq_72.csv
[3/6] Checking FS-Mutant preprocessed task bundle
      found: 3_FsMutant/data/merged.pkl
[4/6] Checking Pichia expression LMDB data
      found: 1_Data/LMDB/Pichia-Pastoris/cls2/train_cv3/fold3/train
[5/6] Checking pretrained PROMEX teacher checkpoint
      found: weights/pretrained/moe_teacher_pretrained_model.pt
[6/6] Checking Pichia mutation CSV
      found: 4_Pichia_Pastoris/4_Fitness_Prediction/data/Pichia-Pastoris/substitutions_round1/single_mutant/DeltaE/Pel114_pectate_lyase.csv
PROMEX demo files OK
```

The tested wall-clock runtime is less than 1 second. Small variation is expected across machines.

## 📦 Demo Data

The one-command artifact check uses real data already included in the release:

| File or directory | Purpose |
| --- | --- |
| `1_Data/LMDB/Stability/normal/train` | Real 2_Property regression LMDB |
| `1_Data/LMDB/Beta-Lactamase/percent1/random1/train` | Real 2_Property few-shot regression LMDB |
| `1_Data/LMDB/Remote-Homology/percent1/random1/train` | Real 2_Property few-shot classification LMDB |
| `1_Data/LMDB/Secondary-Structure/normal/train` | Real 2_Property token-classification LMDB used in teacher adaptation |
| `3_FsMutant/data/struc_sq_72.csv` | Real FS-Mutant structure-sequence table |
| `3_FsMutant/data/fs-mutant/GFP_AEQVI_Sarkisyan_2016/GFP_AEQVI_Sarkisyan_2016.csv` | Real FS-Mutant mutation benchmark CSV |
| `3_FsMutant/data/merged.pkl` | Real preprocessed FS-Mutant task bundle |
| `1_Data/LMDB/Pichia-Pastoris/cls2/train_cv3/fold3/train` | Auxiliary Pichia expression LMDB check |
| `4_Pichia_Pastoris/4_Fitness_Prediction/data/Pichia-Pastoris/substitutions_round1/single_mutant/DeltaE/Pel114_pectate_lyase.csv` | Auxiliary Pichia mutation-candidate CSV |

`demo_data/property_regression_demo.csv` contains six real property-regression rows sampled from Stability, Beta-Lactamase, and Thermostability. It is kept as a small custom-data template for CSV-to-LMDB conversion and is not used by the artifact check.

## 🧬 User Data Format

For property prediction tasks, custom data can be provided as a CSV with three columns. The bundled `demo_data/property_regression_demo.csv` contains six real rows sampled from the property LMDB files:

```text
name,seq,label
stab_train_0,DQSVRKLVRKLPDEGLDREKVKTYLDKLGVDREELQKFSDAIGLESSGGS,-0.209999993443489
stab_train_1,GSSDIEITVEGKEQADKVIEEMKRRNLEVHVEEHNGQYIDKASLESSGGS,-0.949999988079071
beta_train_0,MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLILTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW,0.9426838159561156
beta_train_1,MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTILGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW,0.6457681655883789
Q9NQ94,MESNHKSGDGLSGTQKEAALRALVQRTGYSLVQENGQRKYGGPPPGWDAAPPERGCEIFIGKLPRDLFEDELIPLCEKIGKIYEMRMMMDFNGNNRGYAFVTFSNKVEAKNAIKQLNNYEIRNGRLLGVCASVDNCRLFVGGIPKTKKREEILSEMKKVTEGVVDVIVYPSAADKTKNRGFAFVEYESHRAAAMARRKLLPGRIQLWGHGIAVDWAEPEVEVDEDTMSSVKILYVRNLMLSTSEEMIEKEFNNIKPGAVERVKKIRDYAFVHFSNREDAVEAMKALNGKVLDGSPIEVTLAKPVDKDSYVRYTRGTGGRGTMLQGEYTYSLGQVYDPTTTYLGAPVFYAPQTYAAIPSLHFPATKGHLSNRAIIRAPSVREIYMNVPVGAAGVRGLGGRGYLAYTGLGRGYQVKGDKREDKLYDILPGMELTPMNPVTLKPQGIKLAPQILEEICQKNNWGQPVYQLHSAIGQDQRQLFLYKITIPALASQNPAIHPFTPPKLSAFVDEAKTYAAEYTLQTLGIPTDGGDGTMATAAAAATAFPGYAVPNATAPVSAAQLKQAVTLGQDLAAYTTYEVYPTFAVTARGDGYGTF,41.9455665914228
Q9NRG9,MCSLGLFPPPPPRGQVTLYEHNNELVTGSSYESPPPDFRGQWINLPVLQLTKDPLKTPGRLDHGTRTAFIHHREQVWKRCINIWRDVGLFGVLNEIANSEEEVFEWVKTASGWALALCRWASSLHGSLFPHLSLRSEDLIAEFAQVTNWSSCCLRVFAWHPHTNKFAVALLDDSVRVYNASSTIVPSLKHRLQRNVASLAWKPLSASVLAVACQSCILIWTLDPTSLSTRPSSGCAQVLSHPGHTPVTSLAWAPSGGRLLSASPVDAAIRVWDVSTETCVPLPWFRGGGVTNLLWSPDGSKILATTPSAVFRVWEAQMWTCERWPTLSGRCQTGCWSPDGSRLLFTVLGEPLIYSLSFPERCGEGKGCVGGAKSATIVADLSETTIQTPDGEERLGGEAHSMVWDPSGERLAVLMKGKPRVQDGKPVILLFRTRNSPVFELLPCGIIQGEPGAQPQLITFHPSFNKGALLSVGWSTGRIAHIPLYFVNAQFPRFSPVLGRAQEPPAGGGGSIHDLPLFTETSPTSAPWDPLPGPPPVLPHSPHSHL,51.7216531502637
```

Column meaning:

| Column | Meaning |
| --- | --- |
| `name` | Protein or variant identifier |
| `seq` | SaProt-style sequence string used by the tokenizer |
| `label` | Regression value, classification class ID, or token-label list |

Convert the CSV into the LMDB format used by PROMEX:

```bash
python tools/csv_to_lmdb.py \
  --input demo_data/property_regression_demo.csv \
  --output /tmp/promex_demo_lmdb \
  --overwrite
```

Expected conversion output:

```text
Wrote 6 records to /tmp/promex_demo_lmdb
```

The generated LMDB stores:

```text
length -> number of records
0, 1, 2, ... -> JSON records with name, seq, label
```

For example:

```json
{"name": "stab_train_0", "seq": "DQSVRKLVRKLPDEGLDREKVKTYLDKLGVDREELQKFSDAIGLESSGGS", "label": -0.209999993443489}
```

After conversion, set the `train_lmdb`, `valid_lmdb`, and `test_lmdb` fields in the YAML config files to the new LMDB directories.

For the *Pichia pastoris* expression MetaDistill workflow, the YAML files in `4_Pichia_Pastoris/2_MetaDistill_Expression/config/` expect binary-classification LMDB datasets. Each record follows the same basic shape as the property LMDB files: `name`, `seq`, and binary `label` (`0` or `1`). The prepared LMDB paths are under `1_Data/LMDB/Pichia-Pastoris/cls2/...`.

For the *Pichia pastoris* fitness/ranking workflow, custom target files are plain CSV files placed under the `target_raw_data_dir` configured in `4_Pichia_Pastoris/4_Fitness_Prediction/src/config_round*.json`. The active config is selected in `4_Pichia_Pastoris/4_Fitness_Prediction/src/__init__.py`. The following rows are copied from the real bundled file `4_Pichia_Pastoris/4_Fitness_Prediction/data/Pichia-Pastoris/substitutions_round1/single_mutant/DeltaE/Pel114_pectate_lyase.csv`:

```text
mutant,mutated_sequence,DMS_score
S2T,ATLFSDTFEDGQADGWETQYGSWSVVTVKGGYAYQQSALDKEARASAGSTDWTDYRVEADLNVLDFNGSNRAMLAGRYIDGNNYYAVSLTGGEKLELRKKVRGSSTTLVSKDYPMSEGTAYRVALAAAGSELKVYINGSLELSAADSELKAGRVGLIGYKTAVQFDNVTVAGAGAEVPGGSEPAPTPEPTPEPTPEPTPEPTPEPTPEPTPAPAPVLQSNYDLTGFAAGTTGGGNIGETNAAYKKVYTASDLAAALKKGSGVKVIEIMNDLNLGWNEIPSAAKTAPFSTHNTPLTHPVLLKTGVSKIAIEGFNGLTIFSTNGAKLKHASFTIKRSSNVIIRNLEFDELWEWDEATKGDYDRNDWDYITVEASSKIWIDHCTFNKAYDGLVDVKKGSNGVTISWSVFRGDNQSSTGWVAQQINAMEGSRSSYPMYNYLRSLGLSKEDIIAVAAGQKKGHLIGATEFAVDNANLEVTLHHNYYKDIQDRMPRLRGGNVHVYNIVMDSAGTRASKKRLTSKISSAIASKGYHFGVTSNGAISTEGGALLLENSEIIDVASPVRNNQASASNAAYTGKIKLVNTIYTLDGNTFRGGSEDAGSPLSPKPAAVKAFAWNGFDTLPYTYKAEDPSGLKAQLTGSNGAGAGQLGWSKSSWQITKY,3.294676445618336
S2A,AALFSDTFEDGQADGWETQYGSWSVVTVKGGYAYQQSALDKEARASAGSTDWTDYRVEADLNVLDFNGSNRAMLAGRYIDGNNYYAVSLTGGEKLELRKKVRGSSTTLVSKDYPMSEGTAYRVALAAAGSELKVYINGSLELSAADSELKAGRVGLIGYKTAVQFDNVTVAGAGAEVPGGSEPAPTPEPTPEPTPEPTPEPTPEPTPEPTPAPAPVLQSNYDLTGFAAGTTGGGNIGETNAAYKKVYTASDLAAALKKGSGVKVIEIMNDLNLGWNEIPSAAKTAPFSTHNTPLTHPVLLKTGVSKIAIEGFNGLTIFSTNGAKLKHASFTIKRSSNVIIRNLEFDELWEWDEATKGDYDRNDWDYITVEASSKIWIDHCTFNKAYDGLVDVKKGSNGVTISWSVFRGDNQSSTGWVAQQINAMEGSRSSYPMYNYLRSLGLSKEDIIAVAAGQKKGHLIGATEFAVDNANLEVTLHHNYYKDIQDRMPRLRGGNVHVYNIVMDSAGTRASKKRLTSKISSAIASKGYHFGVTSNGAISTEGGALLLENSEIIDVASPVRNNQASASNAAYTGKIKLVNTIYTLDGNTFRGGSEDAGSPLSPKPAAVKAFAWNGFDTLPYTYKAEDPSGLKAQLTGSNGAGAGQLGWSKSSWQITKY,3.525697111001676
```

`mutant` is 1-indexed mutation notation; multiple sites are separated by `:`. `mutated_sequence` is the full mutated sequence. `DMS_score` is the measured label when available. Run `python preprocess.py` from `4_Pichia_Pastoris/4_Fitness_Prediction`, then follow `4_Pichia_Pastoris/4_Fitness_Prediction/run.sh`.

## 📄 License

The code is released under the MIT License. See `LICENSE`.
