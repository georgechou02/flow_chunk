from types import SimpleNamespace

import pytest

from lerobot.common.wandb_utils import infer_wandb_data_regime


def make_cfg(
    *,
    repo_id: str = "HuggingFaceVLA/libero",
    episodes: list[int] | None = None,
    job_name: str = "experiment",
    output_dir: str = "outputs/train/experiment",
    group: str | None = None,
    configured_regime: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(repo_id=repo_id, episodes=episodes),
        job_name=job_name,
        output_dir=output_dir,
        wandb=SimpleNamespace(group=group, data_regime=configured_regime),
    )


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        (make_cfg(job_name="my_fulldata_run"), "full"),
        (make_cfg(job_name="my_lowdata1_run", episodes=list(range(40))), "lowdata1"),
        (make_cfg(job_name="my_lowdata10_run", episodes=list(range(400))), "lowdata10"),
        (make_cfg(episodes=list(range(40))), "lowdata1"),
        (make_cfg(episodes=list(range(400))), "lowdata10"),
        (make_cfg(), "full"),
        (make_cfg(repo_id="organization/other", configured_regime="lowdata10"), "lowdata10"),
        (make_cfg(repo_id="organization/other"), None),
        (make_cfg(episodes=list(range(100))), None),
    ],
)
def test_infer_wandb_data_regime(cfg: SimpleNamespace, expected: str | None) -> None:
    assert infer_wandb_data_regime(cfg) == expected


def test_infer_wandb_data_regime_rejects_unknown_explicit_value() -> None:
    cfg = make_cfg(configured_regime="small")

    with pytest.raises(ValueError, match="wandb.data_regime"):
        infer_wandb_data_regime(cfg)
