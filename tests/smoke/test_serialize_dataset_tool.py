import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile


def test_serialize_dataset_tool_writes_npy_and_marker(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    input_root = tmp_path / "raw"
    output_root = tmp_path / "serialized"

    src_dir = input_root / "Landsat_01"
    src_dir.mkdir(parents=True, exist_ok=True)
    array = np.arange(16, dtype=np.uint16).reshape(4, 4)
    tifffile.imwrite(src_dir / "Group_01_L_0001.tif", array)

    cmd = [
        sys.executable,
        "tools/serialize_dataset.py",
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
        "--source-suffix",
        ".tif",
        "--format",
        "npy",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    out_file = output_root / "Landsat_01" / "Group_01_L_0001.npy"
    assert out_file.exists()
    loaded = np.load(out_file, allow_pickle=False)
    np.testing.assert_array_equal(loaded, array)

    marker_path = output_root / ".stf_serialized.json"
    assert marker_path.exists()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["data_suffix"] == ".npy"


def test_serialize_dataset_tool_default_sibling_convention(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    input_root = tmp_path / "CIA" / "private_data" / "hh_setting-1-patch"
    for split in ("train", "val"):
        src_dir = input_root / split / "Landsat_01"
        src_dir.mkdir(parents=True, exist_ok=True)
        array = np.arange(9, dtype=np.uint16).reshape(3, 3)
        tifffile.imwrite(src_dir / "Group_01_L_0001.tif", array)

    cmd = [
        sys.executable,
        "tools/serialize_dataset.py",
        "--input-root",
        str(input_root),
        "--splits",
        "train,val",
        "--format",
        "npy",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    output_base = input_root.parent / "hh_setting-1-patch_serialized"
    train_file = output_base / "train" / "Landsat_01" / "Group_01_L_0001.npy"
    val_file = output_base / "val" / "Landsat_01" / "Group_01_L_0001.npy"
    assert train_file.exists()
    assert val_file.exists()
    assert (output_base / "train" / ".stf_serialized.json").exists()
    assert (output_base / "val" / ".stf_serialized.json").exists()
