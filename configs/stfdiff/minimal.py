from functools import partial

import torch
from torch.utils.data import DataLoader

from stf.config.types import DataConfig, ExperimentConfig, IOConfig, TrainConfig
from stf.data import EpochBasedSampler, SpatioTemporalFusionDataset
from stf.data.transforms import Format, LoadData, RescaleToMinusOneOne
from stf.metrics import CC, ERGAS, MAE, PSNR, RMSE, SAM, SSIM, TRP, UIQI
from stf.models import GaussianDiffusion, PredNoiseNet


train_dataset = SpatioTemporalFusionDataset(
    dataset_name="toy",
    data_root="data/toy/formal/train",
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
        LoadData(key_list=["fine_img_01", "fine_img_02", "coarse_img_01", "coarse_img_02"]),
        RescaleToMinusOneOne(key_list=["fine_img_01", "fine_img_02", "coarse_img_01", "coarse_img_02"], data_range=[0, 10000]),
        Format(key_list=["fine_img_01", "fine_img_02", "coarse_img_01", "coarse_img_02"]),
    ],
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    sampler=EpochBasedSampler(dataset=train_dataset, is_shuffle=True, seed=42),
    num_workers=0,
)

val_dataset = SpatioTemporalFusionDataset(
    dataset_name="toy",
    data_root="data/toy/formal/val",
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
        LoadData(key_list=["fine_img_01", "fine_img_02", "coarse_img_01", "coarse_img_02"]),
        RescaleToMinusOneOne(key_list=["fine_img_01", "fine_img_02", "coarse_img_01", "coarse_img_02"], data_range=[0, 10000]),
        Format(key_list=["fine_img_01", "fine_img_02", "coarse_img_01", "coarse_img_02"]),
    ],
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=2,
    sampler=EpochBasedSampler(dataset=val_dataset, is_shuffle=False, seed=42),
    num_workers=0,
)

model = GaussianDiffusion(
    model=PredNoiseNet(dim=64, channels=3, out_dim=3, dim_mults=(1, 2, 4)),
    image_size=256,
    timesteps=100,
    sampling_timesteps=50,
    objective="pred_x0",
    ddim_sampling_eta=0.0,
)

EXPERIMENT = ExperimentConfig(
    task="stfdiff",
    name="minimal",
    model=model,
    optimizer=partial(torch.optim.Adam, lr=1e-4),
    scheduler=partial(
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        mode="min",
        factor=0.5,
        patience=5,
    ),
    metrics=[RMSE(), MAE(), PSNR(max_value=1.0), SSIM(data_range=1.0), ERGAS(ratio=1.0 / 16.0), CC(), SAM(), UIQI(), TRP()],
    data=DataConfig(train_dataloader=train_loader, val_dataloader=val_loader),
    train=TrainConfig(max_epochs=1, val_interval=1, save_interval=1, use_ema=True, use_mixed_precision=True),
    io=IOConfig(output_root="runs", save_images=False, show_images=False, show_bands=(2, 1, 0)),
    resume_from=None,
)
