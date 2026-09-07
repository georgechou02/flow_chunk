#!/usr/bin/env python

from unittest.mock import Mock

import pytest

pytest.importorskip("accelerate", reason="Training requires lerobot[training]")
pytest.importorskip("datasets", reason="Training requires lerobot[training]")

from lerobot.common.wandb_utils import WandBLogger  # noqa: E402
from lerobot.scripts.lerobot_train import _log_per_step_physical_metrics  # noqa: E402


def test_physical_metrics_are_logged_between_regular_log_steps():
    wandb_logger = object.__new__(WandBLogger)
    wandb_logger._wandb = Mock()
    wandb_logger._wandb_custom_step_key = None

    _log_per_step_physical_metrics(
        wandb_logger,
        {
            "physical_loss": 1.25,
            "weighted_physical_loss": 0.5,
            "flow_loss": 3.0,
        },
        step=7,
        is_log_step=False,
    )

    wandb_logger._wandb.log.assert_called_once_with(
        data={"train/physical_loss": 1.25, "train/weighted_physical_loss": 0.5},
        step=7,
    )


def test_physical_metrics_are_not_duplicated_on_regular_log_steps():
    wandb_logger = Mock()

    _log_per_step_physical_metrics(
        wandb_logger,
        {"physical_loss": 1.25, "weighted_physical_loss": 0.5},
        step=100,
        is_log_step=True,
    )

    wandb_logger.log_dict.assert_not_called()


def test_clean_action_metrics_are_logged_on_their_sparse_step():
    wandb_logger = object.__new__(WandBLogger)
    wandb_logger._wandb = Mock()
    wandb_logger._wandb_custom_step_key = None

    _log_per_step_physical_metrics(
        wandb_logger,
        {
            "clean_action_mse": 0.125,
            "clean_action_valid_ratio": 0.75,
            "flow_loss": 3.0,
        },
        step=100,
        is_log_step=False,
    )

    wandb_logger._wandb.log.assert_called_once_with(
        data={"train/clean_action_mse": 0.125, "train/clean_action_valid_ratio": 0.75},
        step=100,
    )
