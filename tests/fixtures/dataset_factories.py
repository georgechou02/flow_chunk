# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

import pandas as pd


def build_annotation_dataset(
    root: Path,
    episode_specs: list[tuple[int, int, str]],
    *,
    fps: int = 10,
) -> Path:
    """Build a minimal LeRobot-shaped dataset on disk for annotation tests.

    ``episode_specs`` is a list of ``(episode_index, num_frames, task_text)``.
    Each episode is written to its own
    ``data/chunk-000/file-{ep:03d}.parquet`` so the writer's per-shard
    rewrite path is exercised. The dataset carries the minimum
    ``meta/tasks.parquet`` + ``meta/info.json`` the reader / executor need;
    it has no videos, so the modules fall back to text-only prompts.

    Shared by the annotation-pipeline pytest fixtures (``tests/annotations/
    conftest.py``) and the opt-in E2E smoke run so the fixture shape lives
    in exactly one place.
    """
    from lerobot.datasets.io_utils import write_tasks
    from lerobot.utils.io_utils import write_json

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    tasks: dict[int, str] = {}
    for episode_index, num_frames, task_text in episode_specs:
        if task_text not in tasks.values():
            tasks[len(tasks)] = task_text
        task_index = next(k for k, v in tasks.items() if v == task_text)
        frame = pd.DataFrame(
            {
                "episode_index": [episode_index] * num_frames,
                "frame_index": list(range(num_frames)),
                "timestamp": [round(i / fps, 6) for i in range(num_frames)],
                "task_index": [task_index] * num_frames,
                "subtask_index": [0] * num_frames,  # legacy column the writer must drop
            }
        )
        frame.to_parquet(data_dir / f"file-{episode_index:03d}.parquet", index=False)

    # Canonical tasks frame: indexed by task string with a ``task_index``
    # column, matching what ``lerobot.datasets.io_utils.load_tasks`` expects.
    tasks_df = pd.DataFrame(
        {"task_index": list(tasks.keys())},
        index=pd.Index(list(tasks.values()), name="task"),
    )
    write_tasks(tasks_df, root)

    write_json(
        {
            "codebase_version": "v3.1",
            "fps": fps,
            "features": {},
            "total_episodes": len(episode_specs),
        },
        root / "meta" / "info.json",
    )
    return root
