from pathlib import Path

from stf.compat import migrate_legacy_config, parse_epoch_from_path


def test_parse_epoch_from_path():
    assert parse_epoch_from_path("/tmp/model_epoch_77.pth") == 77
    assert parse_epoch_from_path("/tmp/model_latest.pth") is None


def test_migrate_legacy_config(tmp_path: Path):
    legacy_cfg = tmp_path / "legacy_flow.py"
    legacy_cfg.write_text(
        "train_dataloader=None\n"
        "val_dataloader=None\n"
        "model=object()\n"
        "optimizer=lambda params: None\n"
        "metric_list=[]\n"
        "MAX_EPOCH=1\n"
        "VAL_INTERVAL=1\n"
        "SAVE_INTERVAL=1\n"
    )

    output_cfg = tmp_path / "migrated.py"
    report = migrate_legacy_config(legacy_cfg, output_cfg)

    assert report["task"] == "flow"
    assert output_cfg.exists()
    content = output_cfg.read_text()
    assert "EXPERIMENT = ExperimentConfig(" in content
    assert str(legacy_cfg) in content
