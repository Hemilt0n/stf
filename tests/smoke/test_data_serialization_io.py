import json

import numpy as np
import tifffile

from stf.data import SpatioTemporalFusionDataset
from stf.data.transforms import LoadData


def test_load_data_supports_tif_npy_npz(tmp_path):
    array = (np.arange(24, dtype=np.uint16).reshape(2, 3, 4))

    tif_path = tmp_path / "sample.tif"
    npy_path = tmp_path / "sample.npy"
    npz_path = tmp_path / "sample.npz"
    tifffile.imwrite(tif_path, array)
    np.save(npy_path, array, allow_pickle=False)
    np.savez_compressed(npz_path, array=array)

    loader = LoadData(key_list=[])
    np.testing.assert_array_equal(loader.load_data(str(tif_path)), array)
    np.testing.assert_array_equal(loader.load_data(str(npy_path)), array)
    np.testing.assert_array_equal(loader.load_data(str(npz_path)), array)


def test_dataset_prefers_marker_suffix(tmp_path):
    data_root = tmp_path / "data"
    for sub in ("Landsat_01", "Landsat_02", "MODIS_01", "MODIS_02"):
        (data_root / sub).mkdir(parents=True, exist_ok=True)

    file_stem_l = "Group_01_L_0001"
    file_stem_m = "Group_01_M_0001"
    for sub, stem in (
        ("Landsat_01", file_stem_l),
        ("Landsat_02", file_stem_l),
        ("MODIS_01", file_stem_m),
        ("MODIS_02", file_stem_m),
    ):
        (data_root / sub / f"{stem}.tif").write_bytes(b"tif")
        (data_root / sub / f"{stem}.npy").write_bytes(b"npy")

    marker = {
        "format": "stf_serialized_dataset",
        "version": 1,
        "data_suffix": ".npy",
    }
    (data_root / ".stf_serialized.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )

    dataset = SpatioTemporalFusionDataset(
        dataset_name="toy",
        data_root=data_root,
        data_prefix_tmpl_dict={
            "fine_img_01": "Landsat_01",
            "fine_img_02": "Landsat_02",
            "coarse_img_01": "MODIS_01",
            "coarse_img_02": "MODIS_02",
        },
        data_name_tmpl_dict={
            "fine_img_01": "{}_L_{}",
            "fine_img_02": "{}_L_{}",
            "coarse_img_01": "{}_M_{}",
            "coarse_img_02": "{}_M_{}",
        },
        transform_func_list=[],
    )

    assert len(dataset.data_path_list) == 1
    sample = dataset.data_path_list[0]
    assert sample["fine_img_01_path"].endswith(".npy")
    assert sample["fine_img_02_path"].endswith(".npy")
    assert sample["coarse_img_01_path"].endswith(".npy")
    assert sample["coarse_img_02_path"].endswith(".npy")
