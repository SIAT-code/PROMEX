import json
import sys
import time
from pathlib import Path
from typing import Any

import lmdb
import pandas as pd
import torch
import transformers


ROOT = Path(__file__).resolve().parent


def read_lmdb_first_record(path: Path) -> tuple[int, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            length_value = txn.get(b"length")
            record_value = txn.get(b"0")
            if length_value is None:
                raise KeyError(f"Missing 'length' key in {path}")
            if record_value is None:
                raise KeyError(f"Missing record key '0' in {path}")
            return int(length_value.decode()), json.loads(record_value.decode())
    finally:
        env.close()


def label_type(label: Any) -> str:
    if isinstance(label, list):
        return f"list[{len(label)}]"
    return type(label).__name__


def summarize_lmdb(label: str, path: Path) -> None:
    length, record = read_lmdb_first_record(path)
    keys = ",".join(sorted(record.keys()))
    name = record.get("name", "NA")
    seq_len = len(record.get("seq", ""))
    print(
        f"{label}: length={length}; keys={keys}; "
        f"first_name={name}; seq_len={seq_len}; label_type={label_type(record.get('label'))}"
    )


def count_existing(paths: list[Path]) -> tuple[int, int]:
    return sum(path.exists() for path in paths), len(paths)


def main() -> None:
    started = time.time()

    print("PROMEX real-data smoke test")
    print(f"python: {'.'.join(map(str, sys.version_info[:3]))}")
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    print()
    print("[2_Property real LMDB]")
    property_lmdbs = [
        (
            "Stability normal train",
            ROOT / "1_Data" / "LMDB" / "Stability" / "normal" / "train",
        ),
        (
            "Beta-Lactamase percent1 random1 train",
            ROOT / "1_Data" / "LMDB" / "Beta-Lactamase" / "percent1" / "random1" / "train",
        ),
        (
            "Remote-Homology percent1 random1 train",
            ROOT / "1_Data" / "LMDB" / "Remote-Homology" / "percent1" / "random1" / "train",
        ),
        (
            "Secondary-Structure normal train",
            ROOT / "1_Data" / "LMDB" / "Secondary-Structure" / "normal" / "train",
        ),
    ]
    for label, lmdb_path in property_lmdbs:
        summarize_lmdb(label, lmdb_path)

    stage1_ckpts = [
        ROOT / "2_Property" / "1_Teacher" / "weights" / "Stability-adapt" / "esm2_t33_650M_UR50D.pt",
        ROOT / "2_Property" / "1_Teacher" / "weights" / "Beta-Lactamase-adapt" / "esm2_t33_650M_UR50D.pt",
        ROOT / "2_Property" / "1_Teacher" / "weights" / "Secondary-Structure-adapt" / "esm2_t33_650M_UR50D.pt",
    ]
    metadistill_ckpts = [
        ROOT / "2_Property" / "2_MetaDistill" / "weights" / task / "normal" / "esm2_t33_650M_UR50D.pt"
        for task in ["Stability", "Thermostability", "Beta-Lactamase", "Remote-Homology"]
    ]
    found, total = count_existing(stage1_ckpts)
    print(f"stage1_teacher_adapt_checkpoints: {found}/{total}")
    found, total = count_existing(metadistill_ckpts)
    print(f"metadistill_normal_checkpoints: {found}/{total}")

    print()
    print("[3_FsMutant real data]")
    fs_structure_csv = ROOT / "3_FsMutant" / "data" / "struc_sq_72.csv"
    fs_structure = pd.read_csv(fs_structure_csv)
    print(
        f"structure_csv: rows={len(fs_structure)}; "
        f"columns={','.join(fs_structure.columns)}"
    )

    fs_mutant_csv = (
        ROOT
        / "3_FsMutant"
        / "data"
        / "fs-mutant"
        / "GFP_AEQVI_Sarkisyan_2016"
        / "GFP_AEQVI_Sarkisyan_2016.csv"
    )
    fs_mutants = pd.read_csv(fs_mutant_csv)
    print(
        "GFP_AEQVI_Sarkisyan_2016: "
        f"rows={len(fs_mutants)}; columns={','.join(fs_mutants.columns[:4])}"
    )

    fs_merged = torch.load(
        ROOT / "3_FsMutant" / "data" / "merged.pkl",
        map_location="cpu",
        weights_only=False,
    )
    fs_keys = list(fs_merged.keys())[:3] if hasattr(fs_merged, "keys") else []
    print(f"merged_pkl_tasks: {len(fs_merged)}; sample={','.join(fs_keys)}")
    print(
        "retrieved_topk_exists: "
        f"{(ROOT / '3_FsMutant' / 'retrieved' / 'topk_esm2_cosine.pkl').exists()}"
    )

    print()
    print("[4_Pichia_Pastoris auxiliary check]")
    summarize_lmdb(
        "Pichia cls2 round1 train",
        ROOT / "1_Data" / "LMDB" / "pichia_pastoris" / "cls2" / "train_cv3" / "fold3" / "train",
    )
    pichia_csv = (
        ROOT
        / "4_Pichia_Pastoris"
        / "3_Fitness_Prediction"
        / "data"
        / "pichia_pastoris"
        / "substitutions_round2"
        / "combinated_mutant"
        / "Protein_12.csv"
    )
    pichia_candidates = pd.read_csv(pichia_csv)
    print(
        f"Pichia Protein_12 candidates: rows={len(pichia_candidates)}; "
        f"columns={','.join(pichia_candidates.columns)}"
    )

    esm_config = ROOT / "weights" / "esm2_t33_650M_UR50D" / "config.json"
    teacher_ckpt = ROOT / "weights" / "pretrained" / "teacher_pretrained_model.pt"
    print()
    print("[artifacts]")
    print(f"esm_config_exists: {esm_config.exists()}")
    print(f"pretrained_teacher_checkpoint_exists: {teacher_ckpt.exists()}")
    print(f"runtime_seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
