import dataclasses
import os
import pathlib
from unittest import mock

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)


class TestAutoConfigureFsdp:
    def _make_config(self, *, batch_size: int = 32, fsdp_devices: int = 1) -> _config.TrainConfig:
        return dataclasses.replace(
            _config._CONFIGS_DICT["debug"],  # noqa: SLF001
            batch_size=batch_size,
            fsdp_devices=fsdp_devices,
        )

    def test_single_device_unchanged(self):
        config = self._make_config()
        with mock.patch("jax.device_count", return_value=1):
            result = train._auto_configure_fsdp(config)
        assert result.fsdp_devices == 1

    def test_fsdp_already_set_unchanged(self):
        config = self._make_config(fsdp_devices=2)
        with mock.patch("jax.device_count", return_value=4):
            result = train._auto_configure_fsdp(config)
        assert result.fsdp_devices == 2

    def test_multi_device_16gb_enables_fsdp_and_scales_batch(self):
        config = self._make_config(batch_size=32)
        with (
            mock.patch("jax.device_count", return_value=4),
            mock.patch.object(train, "_get_per_device_memory_gb", return_value=16.0),
        ):
            result = train._auto_configure_fsdp(config)
        assert result.fsdp_devices == 4
        assert result.batch_size == 8  # 2 samples/device * 4 devices

    def test_multi_device_20gb_scales_batch_moderately(self):
        config = self._make_config(batch_size=32)
        with (
            mock.patch("jax.device_count", return_value=4),
            mock.patch.object(train, "_get_per_device_memory_gb", return_value=20.0),
        ):
            result = train._auto_configure_fsdp(config)
        assert result.fsdp_devices == 4
        assert result.batch_size == 16  # 4 samples/device * 4 devices

    def test_multi_device_high_memory_no_fsdp(self):
        config = self._make_config()
        with (
            mock.patch("jax.device_count", return_value=4),
            mock.patch.object(train, "_get_per_device_memory_gb", return_value=40.0),
        ):
            result = train._auto_configure_fsdp(config)
        assert result.fsdp_devices == 1

    def test_batch_size_adjusted_if_not_divisible(self):
        config = self._make_config(batch_size=10)
        with (
            mock.patch("jax.device_count", return_value=4),
            mock.patch.object(train, "_get_per_device_memory_gb", return_value=22.0),
        ):
            result = train._auto_configure_fsdp(config)
        assert result.fsdp_devices == 4
        assert result.batch_size == 8  # 10 // 4 * 4

    def test_unknown_memory_warns(self):
        config = self._make_config()
        with (
            mock.patch("jax.device_count", return_value=4),
            mock.patch.object(train, "_get_per_device_memory_gb", return_value=None),
        ):
            result = train._auto_configure_fsdp(config)
        assert result.fsdp_devices == 1
