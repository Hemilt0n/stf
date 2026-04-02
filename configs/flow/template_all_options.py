from functools import partial

import torch
from torch.utils.data import DataLoader

from stf.config.types import DataConfig, ExperimentConfig, IOConfig, TrainConfig
from stf.data import EpochBasedSampler, SpatioTemporalFusionDataset
from stf.data.transforms import Format, LoadData, RescaleToMinusOneOne
from stf.metrics import CC, ERGAS, MAE, PSNR, RMSE, SAM, SSIM, TRP, UIQI
from stf.models import GaussianFlowMatching, PredTrajNet


# Full options template:
# - Include current commonly used config knobs (model/data/train/io).
# - Keep performance options ON by default (aligned with change_aware_perf_24g.py).
# - Keep debug options OFF by default.
# - Copy this file to start new experiments on master/new branches.

KEYS = ["fine_img_01", "fine_img_02", "coarse_img_01", "coarse_img_02"]

# Data root can point to raw or serialized directory.
# Example raw:
#   data/CIA/private_data/hh_setting-1-patch/{train,val,test}
# Example serialized:
#   data/CIA/private_data/hh_setting-1-patch_serialized/{train,val,test}
DATASET_ROOT = "data/toy/formal"

train_dataset = SpatioTemporalFusionDataset(
    dataset_name="toy",
    data_root=f"{DATASET_ROOT}/train",
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
    # Dataset-level meta serialization (independent from offline raster serialization).
    is_serialize_data=True,
    transform_func_list=[
        LoadData(key_list=KEYS),
        RescaleToMinusOneOne(key_list=KEYS, data_range=[0, 10000]),
        Format(key_list=KEYS),
    ],
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    sampler=EpochBasedSampler(dataset=train_dataset, is_shuffle=True, seed=42),
    # Performance-related dataloader knobs (ON by default):
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

val_dataset = SpatioTemporalFusionDataset(
    dataset_name="toy",
    data_root=f"{DATASET_ROOT}/val",
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
    is_serialize_data=True,
    transform_func_list=[
        LoadData(key_list=KEYS),
        RescaleToMinusOneOne(key_list=KEYS, data_range=[0, 10000]),
        Format(key_list=KEYS),
    ],
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=2,
    sampler=EpochBasedSampler(dataset=val_dataset, is_shuffle=False, seed=42),
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

model = GaussianFlowMatching(
    model=PredTrajNet(dim=64, channels=3, out_dim=3, dim_mults=(1, 2, 4)),
    # Base model knobs:
    loss_type="l1",
    num_steps=20,
    noise_std=1.0,
    path_schedule="linear",
    path_power=1.0,
    # Optional objective knobs (kept neutral by default):
    volume_consistency_weight=0.0,
    condition_dropout_p=0.0,
    change_loss_weight=0.0,
    coarse_consistency_weight=0.0,
    coarse_consistency_loss_type="l1",
)

EXPERIMENT = ExperimentConfig(
    # task now supports arbitrary string; recommended convention remains "flow"/"stfdiff".
    task="flow",
    name="template_all_options",
    model=model,
    optimizer=partial(torch.optim.Adam, lr=2e-5),
    scheduler=None,
    metrics=[
        RMSE(),
        MAE(),
        PSNR(max_value=1.0),
        SSIM(data_range=1.0),
        ERGAS(ratio=1.0 / 16.0),
        CC(),
        SAM(),
        UIQI(),
        TRP(),
    ],
    data=DataConfig(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=None,
    ),
    train=TrainConfig(
        # Base training knobs:
        max_epochs=1,
        val_interval=1,
        save_interval=1,
        grad_clip_norm=1.0,
        use_ema=True,
        train_log_interval=1,
        # Performance knobs (ON by default; aligned with change_aware_perf_24g.py):
        use_mixed_precision=True,
        precision="bf16",
        enable_tf32=True,
        deterministic=False,
        cudnn_benchmark=True,
        non_blocking_transfer=True,
        compile_model=False,
        compile_mode="max-autotune",
        compile_dynamic=False,
        use_channels_last=True,
        # Optional fine_t1 noise warmup knobs (OFF by default):
        fine_t1_noise_warmup_epochs=0,
        fine_t1_noise_warmup_steps=0,
        fine_t1_noise_power=4.0,
        fine_t1_noise_std=1.0,
        fine_t1_noise_alpha_tail=0.0,
        # Debug/observability knobs (OFF by default):
        val_step_log_keys=False,
        val_step_log_max_keys=8,
        val_step_save_csv=False,
    ),
    io=IOConfig(
        output_root="runs",
        # Debug I/O knobs (OFF by default):
        save_images=False,
        show_images=False,
        show_bands=(2, 1, 0),
    ),
    seed=42,
    resume_from=None,
    message="template config with full options and categorized comments",
    legacy={},
)
