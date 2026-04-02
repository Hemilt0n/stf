from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Union, Any
import copy
import re
import numpy as np
import pickle
import json


def _load_preferred_data_suffix(data_root: Path):
    marker_path = data_root / ".stf_serialized.json"
    if not marker_path.exists():
        return None
    try:
        with marker_path.open("r", encoding="utf-8") as f:
            marker = json.load(f)
    except Exception:
        return None

    suffix = marker.get("data_suffix")
    if suffix is None:
        return None
    suffix = str(suffix).strip()
    if not suffix:
        return None
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return suffix.lower()


class SpatioTemporalFusionDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        data_root: Union[str, Path],
        data_prefix_tmpl_dict: dict,
        data_name_tmpl_dict: dict,
        is_serialize_data: bool = False,
        transform_func_list: List = None,
    ):
        super(SpatioTemporalFusionDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.data_prefix_tmpl_dict = data_prefix_tmpl_dict
        self.data_name_tmpl_dict = data_name_tmpl_dict
        # self.search_key = 'fine_img_01'
        self.search_key = list(data_prefix_tmpl_dict.keys())[0]
        self.preferred_data_suffix = _load_preferred_data_suffix(self.data_root)
        self.data_path_list = self.load_data_list()

        self.is_serialize_data = is_serialize_data
        if self.is_serialize_data:
            self.data_bytes, self.data_address = self.serialize_data()
        self.transform_func_list = transform_func_list

    def serialize_data(self):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        data_bytes_list = [_serialize(data) for data in self.data_path_list]
        if not data_bytes_list:
            return np.asarray([], dtype=np.uint8), np.asarray([], dtype=np.int64)
        data_size_list = np.asarray(
            [len(data_bytes) for data_bytes in data_bytes_list], dtype=np.int64
        )
        data_address = np.cumsum(data_size_list)
        data_types = np.concatenate(data_bytes_list)
        return data_types, data_address

    def load_data_list(self):
        data_path_segment_list = self.get_data_path_segment_list()
        data_path_list = []
        for (
            data_prefix_name_filled_context,
            data_file_name_filled_context,
            data_file_suffix,
        ) in data_path_segment_list:
            data_path_key = (
                '_'.join(data_prefix_name_filled_context)
                + '-'
                + '_'.join(data_file_name_filled_context)
            )
            data_path_dict = dict(key=data_path_key)
            for key in self.data_prefix_tmpl_dict.keys():
                data_prefix_tmpl = self.data_prefix_tmpl_dict[key]
                data_name_tmpl = self.data_name_tmpl_dict[key]
                data_prefix = data_prefix_tmpl.format(*data_prefix_name_filled_context)
                data_name = data_name_tmpl.format(*data_file_name_filled_context)
                path = self.data_root / data_prefix / f'{data_name}{data_file_suffix}'
                data_path_dict[f'{key}_path'] = str(path)
            data_path_list.append(data_path_dict)
        return data_path_list

    def get_data_path_segment_list(self):
        data_path_segment_list = []

        data_prefix_tmpl = self.data_prefix_tmpl_dict[self.search_key]
        data_prefix_regexp_for_pathlib = data_prefix_tmpl.replace('{}', '*')
        data_prefix_regexp_for_re = data_prefix_tmpl.replace('{}', '(.*)')

        data_name_tmpl = self.data_name_tmpl_dict[self.search_key]
        data_name_regexp_for_pathlib = data_name_tmpl.replace('{}', '*')
        data_name_regexp_for_re = data_name_tmpl.replace('{}', '(.*)')

        data_prefix_path_list = sorted(
            list(self.data_root.glob(data_prefix_regexp_for_pathlib))
        )
        for data_prefix_path in data_prefix_path_list:
            data_prefix_name = data_prefix_path.name
            data_prefix_name_filled_context = re.findall(
                data_prefix_regexp_for_re, data_prefix_name
            )[0]
            data_prefix_name_filled_context = (
                data_prefix_name_filled_context
                if isinstance(data_prefix_name_filled_context, tuple)
                else (data_prefix_name_filled_context,)
            )
            data_path_list = sorted(
                list(data_prefix_path.glob(data_name_regexp_for_pathlib))
            )
            for data_path in data_path_list:
                if (
                    self.preferred_data_suffix is not None
                    and data_path.suffix.lower() != self.preferred_data_suffix
                ):
                    continue
                data_file_name, data_file_suffix = (
                    data_path.stem,
                    data_path.suffix,
                )
                data_file_name_filled_context = re.findall(
                    data_name_regexp_for_re, data_file_name
                )[0]
                data_file_name_filled_context = (
                    data_file_name_filled_context
                    if isinstance(data_file_name_filled_context, tuple)
                    else (data_file_name_filled_context,)
                )
                data_path_segment_list.append(
                    (
                        data_prefix_name_filled_context,
                        data_file_name_filled_context,
                        data_file_suffix,
                    )
                )
        return data_path_segment_list

    def __getitem__(self, idx: int) -> Any:
        if self.is_serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(self.data_bytes[start_addr:end_addr])
            data_info = pickle.loads(bytes)
        else:
            data_info = copy.deepcopy(self.data_path_list[idx])
        data_info['sample_idx'] = idx
        data_info['dataset_name'] = self.dataset_name
        for transform_func in self.transform_func_list:
            data_info = transform_func(data_info)
        return data_info

    def __len__(self) -> int:
        if self.is_serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_path_list)


class SpatioTemporalFusionDatasetForSPSTFM(Dataset):
    def __init__(
        self,
        dataset_name,
        data_root: Union[str, Path] = None,
        extend_data_root: Union[str, Path] = None,
        data_prefix_tmpl_dict: dict = {},
        data_name_tmpl_dict: dict = {},
        data_suffix_dcit: dict = {},
        is_serialize_data: bool = False,
        transform_func_list: List = None,
    ):
        super(SpatioTemporalFusionDatasetForSPSTFM, self).__init__()
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.extend_data_root = (
            Path(extend_data_root) if extend_data_root is not None else None
        )
        self.data_prefix_tmpl_dict = data_prefix_tmpl_dict
        self.data_name_tmpl_dict = data_name_tmpl_dict
        self.data_suffix_dcit = data_suffix_dcit
        # self.search_key = 'fine_img_01'
        self.search_key = list(data_prefix_tmpl_dict.keys())[0]
        self.preferred_data_suffix = _load_preferred_data_suffix(self.data_root)
        self.data_path_list = self.load_data_list()

        self.is_serialize_data = is_serialize_data
        if self.is_serialize_data:
            self.data_bytes, self.data_address = self.serialize_data()
        self.transform_func_list = transform_func_list

    def serialize_data(self):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        data_bytes_list = [_serialize(data) for data in self.data_path_list]
        if not data_bytes_list:
            return np.asarray([], dtype=np.uint8), np.asarray([], dtype=np.int64)
        data_size_list = np.asarray(
            [len(data_bytes) for data_bytes in data_bytes_list], dtype=np.int64
        )
        data_address = np.cumsum(data_size_list)
        data_types = np.concatenate(data_bytes_list)
        return data_types, data_address

    def load_data_list(self):
        data_path_segment_list = self.get_data_path_segment_list()
        data_path_list = []
        for (
            data_prefix_name_filled_context,
            data_file_name_filled_context,
            data_file_suffix,
        ) in data_path_segment_list:
            data_path_key = (
                '_'.join(data_prefix_name_filled_context)
                + '-'
                + '_'.join(data_file_name_filled_context)
            )
            data_path_dict = dict(key=data_path_key)
            for key in self.data_prefix_tmpl_dict.keys():
                if key == 'extend_data':
                    for extend_data_key in self.data_prefix_tmpl_dict[key].keys():
                        data_prefix_tmpl = self.data_prefix_tmpl_dict[key][
                            extend_data_key
                        ]
                        data_name_tmpl = self.data_name_tmpl_dict[key][extend_data_key]
                        data_prefix = data_prefix_tmpl.format(
                            *data_prefix_name_filled_context
                        )
                        data_name = data_name_tmpl.format(
                            *data_file_name_filled_context
                        )
                        preset_data_file_suffix = self.data_suffix_dcit.get(
                            extend_data_key, None
                        )
                        data_file_suffix = (
                            preset_data_file_suffix
                            if preset_data_file_suffix is not None
                            else data_file_suffix
                        )
                        path = (
                            self.extend_data_root
                            / data_prefix
                            / f'{data_name}{data_file_suffix}'
                        )
                        data_path_dict[f'{extend_data_key}_path'] = str(path)
                else:
                    data_prefix_tmpl = self.data_prefix_tmpl_dict[key]
                    data_name_tmpl = self.data_name_tmpl_dict[key]
                    data_prefix = data_prefix_tmpl.format(
                        *data_prefix_name_filled_context
                    )
                    data_name = data_name_tmpl.format(*data_file_name_filled_context)
                    preset_data_file_suffix = self.data_suffix_dcit.get(key, None)
                    data_file_suffix = (
                        preset_data_file_suffix
                        if preset_data_file_suffix is not None
                        else data_file_suffix
                    )
                    path = (
                        self.data_root / data_prefix / f'{data_name}{data_file_suffix}'
                    )
                    data_path_dict[f'{key}_path'] = str(path)
            data_path_list.append(data_path_dict)
        return data_path_list

    def get_data_path_segment_list(self):
        data_path_segment_list = []

        data_prefix_tmpl = self.data_prefix_tmpl_dict[self.search_key]
        data_prefix_regexp_for_pathlib = data_prefix_tmpl.replace('{}', '*')
        data_prefix_regexp_for_re = data_prefix_tmpl.replace('{}', '(.*)')

        data_name_tmpl = self.data_name_tmpl_dict[self.search_key]
        data_name_regexp_for_pathlib = data_name_tmpl.replace('{}', '*')
        data_name_regexp_for_re = data_name_tmpl.replace('{}', '(.*)')

        data_prefix_path_list = sorted(
            list(self.data_root.glob(data_prefix_regexp_for_pathlib))
        )
        for data_prefix_path in data_prefix_path_list:
            data_prefix_name = data_prefix_path.name
            data_prefix_name_filled_context = re.findall(
                data_prefix_regexp_for_re, data_prefix_name
            )[0]
            data_prefix_name_filled_context = (
                data_prefix_name_filled_context
                if isinstance(data_prefix_name_filled_context, tuple)
                else (data_prefix_name_filled_context,)
            )
            data_path_list = sorted(
                list(data_prefix_path.glob(data_name_regexp_for_pathlib))
            )
            for data_path in data_path_list:
                if (
                    self.preferred_data_suffix is not None
                    and data_path.suffix.lower() != self.preferred_data_suffix
                ):
                    continue
                data_file_name, data_file_suffix = (
                    data_path.stem,
                    data_path.suffix,
                )
                data_file_name_filled_context = re.findall(
                    data_name_regexp_for_re, data_file_name
                )[0]
                data_file_name_filled_context = (
                    data_file_name_filled_context
                    if isinstance(data_file_name_filled_context, tuple)
                    else (data_file_name_filled_context,)
                )
                data_path_segment_list.append(
                    (
                        data_prefix_name_filled_context,
                        data_file_name_filled_context,
                        data_file_suffix,
                    )
                )
        return data_path_segment_list

    def __getitem__(self, idx: int) -> Any:
        if self.is_serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(self.data_bytes[start_addr:end_addr])
            data_info = pickle.loads(bytes)
        else:
            data_info = copy.deepcopy(self.data_path_list[idx])
        data_info['sample_idx'] = idx
        data_info['dataset_name'] = self.dataset_name
        for transform_func in self.transform_func_list:
            data_info = transform_func(data_info)
        return data_info

    def __len__(self) -> int:
        if self.is_serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_path_list)
