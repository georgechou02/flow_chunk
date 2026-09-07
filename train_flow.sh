#!/usr/bin/env bash
# FlowPolicy first-order-supervision sweep for LIBERO.
#
# FlowPolicy has no task-language conditioning, so every suite is trained as an
# independent policy. By default this launches four runs/checkpoints per seed
# and hyperparameter combination, one for each standard 10-task suite.
#
# Examples:
#   # Train and evaluate only task 0 from LIBERO-10.
#   TRAIN_SUITES=libero_10 TRAIN_TASK_IDS='[0]' \
#   TRAIN_TRAJECTORIES_PER_TASK=5 TRAJECTORY_SELECTION_SEED=1000 \
#   ENV_EVAL_FREQ=100000 EVAL_EPISODES=10 bash train_flow.sh
#
#   # Train only LIBERO-Spatial on four GPUs.
#   TRAIN_SUITES=libero_spatial \
#   GPUS='["0,1,2,3"]' RUN_SEEDS='[1000]' bash train_flow.sh
#   # The default physical-loss sweep is 0.01 and 0.1; pin one value with:
#   # PHY_LOSS_WEIGHT=0.1 ...
#
#   # Train all four suites; available GPU groups consume independent runs.
#   GPUS='["0,1,2,3","4,5,6,7"]' \
#   RUN_SEEDS='[1000]' \
#   INTERPOLATION_MODE=dct \
#   POLICY_DCT_COE_NUM='[24,48]' \
#   bash train_flow.sh
#
#   # B-spline p/M sweep. M < H applies least-squares smoothing.
#   INTERPOLATION_MODE=bspline \
#   BSPLINE_DEGREE='[2,3]' \
#   BSPLINE_COE_NUM='[24,32,48]' \
#   RUN_SEEDS='[1000]' \
#   bash train_flow.sh
#
#   # Train flow-only for 16k global steps, then enable the configured
#   # kinematic loss for the final 4k steps of the same 20k run.
#   TRAIN_SUITES=libero_10 \
#   STEPS=20000 PRE_TRAIN_STEPS=16000 \
#   POLICY_LAMBDA_FLOW_K='[0.01]' \
#   PHY_LOSS_WEIGHT='[0]' \
#   bash train_flow.sh




# TRAIN_SUITES=libero_10 \
# TRAIN_TASK_IDS='[0]' \
# TRAIN_TRAJECTORIES_PER_TASK=5 \
# TRAJECTORY_SELECTION_SEED=1000 \
# GPUS='[2]' \
# STEPS=8000 \
# ENV_EVAL_FREQ=8000 \
# EVAL_EPISODES=50 \
# bash train_flow.sh

set -euo pipefail

cd "$(dirname "$0")"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ "$MUJOCO_GL" == "egl" ]]; then
  export PYOPENGL_PLATFORM="egl"
  if [[ ! -r /usr/share/glvnd/egl_vendor.d/10_nvidia.json && -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    NVIDIA_EGL_VENDOR_FILE="$PWD/nvidia_egl_vendor.json"
    if [[ ! -r "$NVIDIA_EGL_VENDOR_FILE" ]]; then
      echo "Missing NVIDIA EGL vendor file: $NVIDIA_EGL_VENDOR_FILE" >&2
      exit 1
    fi
    export __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL_VENDOR_FILE"
  fi
elif [[ "$MUJOCO_GL" == "osmesa" ]]; then
  export PYOPENGL_PLATFORM="osmesa"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TMPDIR="${TMPDIR:-/data/zhouzhi/tmp}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${TMPDIR}/hf_datasets_cache_flow}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR}/numba_cache}"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$NUMBA_CACHE_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/zhouzhi/conda_envs/lerobot/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data/zhouzhi/conda_envs/lerobot/bin/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-/data/zhouzhi/conda_envs/lerobot/bin/lerobot-train}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi
if [[ ! -x "$ACCELERATE_BIN" ]]; then
  ACCELERATE_BIN="$(command -v accelerate)"
fi
if [[ ! -x "$LEROBOT_TRAIN_BIN" ]]; then
  LEROBOT_TRAIN_BIN="$(command -v lerobot-train)"
fi

# `DATASET_EPISODES`, when supplied, is an additional restriction intersected
# with each suite's automatically selected demonstrations.
LIBERO_ROOT="${LIBERO_ROOT:-/home/zhouzhi/.cache/huggingface/lerobot/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e}"
DATASET_EPISODES="${DATASET_EPISODES:-}"
# TRAIN_SUITES="${TRAIN_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
TRAIN_SUITES="${TRAIN_SUITES:-libero_10}"
# Optional suite-local task ids. When set, both training episodes and online
# evaluation are restricted to these tasks.
TRAIN_TASK_IDS="${TRAIN_TASK_IDS:-}"
# Optional deterministic low-data selection. This is applied after
# TRAIN_TASK_IDS and DATASET_EPISODES have restricted the candidate episodes.
TRAIN_TRAJECTORIES_PER_TASK="${TRAIN_TRAJECTORIES_PER_TASK:-}"
TRAJECTORY_SELECTION_SEED="${TRAJECTORY_SELECTION_SEED:-1000}"

DEFAULT_DECODED_IMAGE_CACHE_ROOT="/home/zhouzhi/.cache/lerobot_decoded/HuggingFaceVLA--libero/86958911c0f959db2bbbdb107eb3e17c5f9c798e"
if [[ -v DECODED_IMAGE_CACHE_ROOT ]]; then
  DECODED_IMAGE_CACHE_ROOT="${DECODED_IMAGE_CACHE_ROOT}"
elif [[ -f "${DEFAULT_DECODED_IMAGE_CACHE_ROOT}/manifest.json" ]]; then
  DECODED_IMAGE_CACHE_ROOT="${DEFAULT_DECODED_IMAGE_CACHE_ROOT}"
else
  DECODED_IMAGE_CACHE_ROOT=""
fi

# A string such as "0,1" means two independent single-GPU jobs. Use a JSON
# list of comma-separated groups for parallel multi-GPU jobs.
GPUS="${GPUS:-${GPU_IDS:-${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0 1}}}}"
RUN_SEEDS="${RUN_SEEDS:-${SEEDS:-${SEED:-1000}}}"

STEPS="${STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-12}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-100}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-false}"

# FlowPolicy architecture.
POLICY_N_OBS_STEPS="${POLICY_N_OBS_STEPS:-2}"
POLICY_HORIZON="${POLICY_HORIZON:-48}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-24}"
POLICY_DROP_N_LAST_FRAMES="${POLICY_DROP_N_LAST_FRAMES:-23}"
POLICY_DOWN_DIMS="${POLICY_DOWN_DIMS:-[512,1024,2048]}"
POLICY_KERNEL_SIZE="${POLICY_KERNEL_SIZE:-5}"
POLICY_N_GROUPS="${POLICY_N_GROUPS:-8}"
POLICY_TIMESTEP_EMBED_DIM="${POLICY_TIMESTEP_EMBED_DIM:-128}"
POLICY_RESIZE_SHAPE="${POLICY_RESIZE_SHAPE:-[256,256]}"
POLICY_CROP_RATIO="${POLICY_CROP_RATIO:-0.875}"
POLICY_CROP_IS_RANDOM="${POLICY_CROP_IS_RANDOM:-true}"
POLICY_VISION_BACKBONE="${POLICY_VISION_BACKBONE:-resnet18}"
POLICY_PRETRAINED_BACKBONE_WEIGHTS="${POLICY_PRETRAINED_BACKBONE_WEIGHTS:-ResNet18_Weights.IMAGENET1K_V1}"
POLICY_USE_GROUP_NORM="${POLICY_USE_GROUP_NORM:-false}"
POLICY_SPATIAL_SOFTMAX_NUM_KEYPOINTS="${POLICY_SPATIAL_SOFTMAX_NUM_KEYPOINTS:-32}"
POLICY_USE_SEPARATE_RGB_ENCODER_PER_CAMERA="${POLICY_USE_SEPARATE_RGB_ENCODER_PER_CAMERA:-false}"
POLICY_USE_FILM_SCALE_MODULATION="${POLICY_USE_FILM_SCALE_MODULATION:-true}"
POLICY_COMPILE_MODEL="${POLICY_COMPILE_MODEL:-false}"

# Flow matching and first-order supervision.
POLICY_SIGMA_MIN="${POLICY_SIGMA_MIN:-0.0}"
POLICY_NUM_INFERENCE_STEPS="${POLICY_NUM_INFERENCE_STEPS:-32}"
POLICY_TIMESTEP_SAMPLING_STRATEGY="${POLICY_TIMESTEP_SAMPLING_STRATEGY:-beta}"
POLICY_TIMESTEP_SAMPLING_ALPHA="${POLICY_TIMESTEP_SAMPLING_ALPHA:-1.5}"
POLICY_TIMESTEP_SAMPLING_BETA="${POLICY_TIMESTEP_SAMPLING_BETA:-1.0}"
POLICY_TIMESTEP_SAMPLING_S="${POLICY_TIMESTEP_SAMPLING_S:-0.999}"
POLICY_LAMBDA_FLOW_K="${POLICY_LAMBDA_FLOW_K:-0.1}"
# During these initial global optimizer steps, the effective kinematic weight
# is zero. The configured lambda remains non-zero so the derivative data
# stencil stays fixed when supervision switches on.
PRE_TRAIN_STEPS="${PRE_TRAIN_STEPS:-${POLICY_PRE_TRAIN_STEPS:-0}}"
PHY_LOSS_WEIGHT="${PHY_LOSS_WEIGHT:-0}"
# Kinematic supervision always includes the action JVP with gradients enabled.
POLICY_SAMPLE_FREQUENCY="${POLICY_SAMPLE_FREQUENCY:-10.0}"
POLICY_GRIPPER_FIRST="${POLICY_GRIPPER_FIRST:-false}"
POLICY_INTERPOLATION_MODE="${POLICY_INTERPOLATION_MODE:-${INTERPOLATION_MODE:-bspline}}"
POLICY_DCT_COE_NUM="${POLICY_DCT_COE_NUM:-48}"
POLICY_BSPLINE_DEGREE="${POLICY_BSPLINE_DEGREE:-${BSPLINE_DEGREE:-2}}"
POLICY_BSPLINE_COE_NUM="${POLICY_BSPLINE_COE_NUM:-${BSPLINE_COE_NUM:-48}}"
POLICY_CONDITIONING_DERIVATIVE_MODE="${POLICY_CONDITIONING_DERIVATIVE_MODE:-central}"
POLICY_IMAGE_ONLY_CONDITION_JVP="${POLICY_IMAGE_ONLY_CONDITION_JVP:-false}"

# Optimization.
POLICY_OPTIMIZER_LR="${POLICY_OPTIMIZER_LR:-1e-4}"
POLICY_OPTIMIZER_WEIGHT_DECAY="${POLICY_OPTIMIZER_WEIGHT_DECAY:-1e-6}"
POLICY_SCHEDULER_WARMUP_STEPS="${POLICY_SCHEDULER_WARMUP_STEPS:-500}"
POLICY_DO_MASK_LOSS_FOR_PADDING="${POLICY_DO_MASK_LOSS_FOR_PADDING:-false}"

ENV_EVAL_FREQ="${ENV_EVAL_FREQ:-20000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
EVAL_USE_ASYNC_ENVS="${EVAL_USE_ASYNC_ENVS:-true}"
EVAL_OBSERVATION_HEIGHT="${EVAL_OBSERVATION_HEIGHT:-256}"
EVAL_OBSERVATION_WIDTH="${EVAL_OBSERVATION_WIDTH:-256}"

RUN_PREFIX="${RUN_PREFIX:-flow-first-order-libero}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
# Opt-in compact output names for targeted follow-up runs. Defaults preserve
# all historical FlowPolicy output paths.
RUN_NAME_STYLE="${RUN_NAME_STYLE:-full}"
APPEND_RUN_TIMESTAMP="${APPEND_RUN_TIMESTAMP:-true}"

WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-flow_first_order_libero}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-${WANDB_RUN_GROUP:-}}"
WANDB_NOTES="${WANDB_NOTES:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

parse_list() {
  local spec="$1"
  local kind="$2"
  LIST_SPEC="$spec" LIST_KIND="$kind" LIST_HORIZON="$POLICY_HORIZON" "$PYTHON_BIN" - <<'PY'
import ast
import math
import os
import re

spec = os.environ["LIST_SPEC"].strip()
kind = os.environ["LIST_KIND"]
horizon = int(os.environ["LIST_HORIZON"])

if not spec:
    raise SystemExit(f"{kind} list is empty")

try:
    parsed = ast.literal_eval(spec)
except Exception:
    parsed = [part for part in re.split(r"[\s,]+", spec.strip("[]()")) if part]

if isinstance(parsed, (str, int, float, bool)):
    values = [parsed]
elif isinstance(parsed, (list, tuple, set)):
    values = list(parsed)
else:
    raise SystemExit(f"Unsupported {kind} list syntax: {spec}")

if kind in {"dct_coe_num", "bspline_degree", "bspline_coe_num"}:
    normalized_values = []
    for value in values:
        text = str(value).strip()
        if not re.fullmatch(r"[0-9]+", text):
            raise SystemExit(f"{kind} must be a non-negative integer, got {value!r}")
        value = int(text)
        if kind == "dct_coe_num" and value > horizon:
            raise SystemExit(f"dct_coe_num must not exceed horizon {horizon}, got {value}")
        if kind == "bspline_degree" and not 2 <= value < horizon:
            raise SystemExit(f"bspline_degree must be in [2, {horizon - 1}], got {value}")
        if kind == "bspline_coe_num" and not 3 <= value <= horizon:
            raise SystemExit(f"bspline_coe_num must be in [3, {horizon}], got {value}")
        normalized_values.append(value)
    values = normalized_values
elif kind in {"lambda_flow_k", "phy_loss_weight"}:
    normalized_values = []
    for value in values:
        text = str(value).strip()
        try:
            numeric_value = float(text)
        except ValueError as exc:
            raise SystemExit(f"{kind} must be a number, got {value!r}") from exc
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise SystemExit(f"{kind} must be finite and non-negative, got {value!r}")
        normalized_values.append(text)
    values = normalized_values
elif kind == "suite":
    allowed = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    normalized_values = []
    for value in values:
        text = str(value).strip().lower()
        if text not in allowed:
            raise SystemExit(f"Unsupported LIBERO suite {value!r}; choose from {sorted(allowed)}")
        if text in normalized_values:
            raise SystemExit(f"Duplicate LIBERO suite {text!r}")
        normalized_values.append(text)
    values = normalized_values

for value in values:
    if kind in {"seed", "task"}:
        value = int(value)
        if value < 0:
            raise SystemExit(f"{kind} must be non-negative")
        print(value)
    elif kind in {
        "lambda_flow_k",
        "phy_loss_weight",
        "dct_coe_num",
        "bspline_degree",
        "bspline_coe_num",
        "suite",
    }:
        print(value)
    elif kind in {"gripper_first", "image_only_condition_jvp"}:
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            print("true")
        elif text in {"0", "false", "no", "n", "off"}:
            print("false")
        else:
            raise SystemExit(f"{kind} must be true or false, got {value!r}")
    else:
        text = str(value).strip()
        if not text:
            raise SystemExit("gpu id must be non-empty")
        print(text)
PY
}

resolve_suite_episodes() {
  local suite="$1"
  LIBERO_DATASET_ROOT="$LIBERO_ROOT" \
  LIBERO_SUITE="$suite" \
  LIBERO_TASK_IDS="$TRAIN_TASK_IDS_JSON" \
  LIBERO_TRAJECTORIES_PER_TASK="$TRAIN_TRAJECTORIES_PER_TASK" \
  LIBERO_TRAJECTORY_SELECTION_SEED="$TRAJECTORY_SELECTION_SEED" \
  MANUAL_EPISODES="$DATASET_EPISODES" \
  "$PYTHON_BIN" - <<'PY'
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import random
import re

import pyarrow.parquet as pq
from libero.libero import benchmark

root = Path(os.environ["LIBERO_DATASET_ROOT"])
suite_name = os.environ["LIBERO_SUITE"]
task_ids_spec = os.environ["LIBERO_TASK_IDS"].strip()
trajectories_per_task_spec = os.environ["LIBERO_TRAJECTORIES_PER_TASK"].strip()
selection_seed = int(os.environ["LIBERO_TRAJECTORY_SELECTION_SEED"])
episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
if not episode_files:
    raise SystemExit(f"No episode metadata found under {root / 'meta' / 'episodes'}")

with redirect_stdout(StringIO()):
    suite = benchmark.get_benchmark_dict()[suite_name]()
if task_ids_spec:
    task_ids = json.loads(task_ids_spec)
    invalid = [task_id for task_id in task_ids if task_id < 0 or task_id >= len(suite.tasks)]
    if invalid:
        raise SystemExit(
            f"Task ids {invalid} are out of range for {suite_name}; "
            f"expected ids in [0, {len(suite.tasks) - 1}]"
        )
    suite_tasks = {suite.get_task(task_id).language.strip() for task_id in task_ids}
else:
    task_ids = list(range(len(suite.tasks)))
suite_task_ids = {suite.get_task(task_id).language.strip(): task_id for task_id in task_ids}
suite_tasks = set(suite_task_ids)

tasks_table = pq.read_table(root / "meta" / "tasks.parquet").to_pylist()
dataset_task_ids = {
    str(row["__index_level_0__"]).strip(): int(row["task_index"])
    for row in tasks_table
}
missing_dataset_task_ids = sorted(suite_tasks - set(dataset_task_ids))
if missing_dataset_task_ids:
    raise SystemExit(f"Dataset task metadata is missing tasks: {missing_dataset_task_ids}")

selected = []
episode_tasks = {}
observed_tasks = set()
all_episode_ids = set()
for path in episode_files:
    table = pq.read_table(path, columns=["episode_index", "tasks"]).to_pydict()
    for episode_index, tasks in zip(table["episode_index"], table["tasks"], strict=True):
        episode_index = int(episode_index)
        if episode_index in all_episode_ids:
            raise SystemExit(f"Duplicate episode_index={episode_index} in dataset metadata")
        all_episode_ids.add(episode_index)
        task_names = {str(task).strip() for task in tasks}
        matched = task_names & suite_tasks
        observed_tasks.update(matched)
        if task_names and task_names <= suite_tasks:
            selected.append(episode_index)
            episode_tasks[episode_index] = task_names

missing_tasks = sorted(suite_tasks - observed_tasks)
if missing_tasks:
    raise SystemExit(
        f"Dataset {root} does not contain all tasks for {suite_name}; missing={missing_tasks}"
    )

manual_spec = os.environ["MANUAL_EPISODES"].strip()
if manual_spec:
    try:
        manual = ast.literal_eval(manual_spec)
    except Exception:
        manual = [part for part in re.split(r"[\s,]+", manual_spec.strip("[]()")) if part]
    if isinstance(manual, int):
        manual = [manual]
    if not isinstance(manual, (list, tuple, set)):
        raise SystemExit(f"DATASET_EPISODES must be an integer list, got {manual_spec!r}")
    try:
        manual_ids = {int(value) for value in manual}
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"DATASET_EPISODES must be an integer list, got {manual_spec!r}") from exc
    unknown = sorted(manual_ids - all_episode_ids)
    if unknown:
        raise SystemExit(f"DATASET_EPISODES contains unknown episode ids: {unknown}")
    selected = [episode_index for episode_index in selected if episode_index in manual_ids]

if trajectories_per_task_spec:
    trajectories_per_task = int(trajectories_per_task_spec)
    sampled = []
    for task_name in sorted(suite_tasks, key=suite_task_ids.get):
        candidates = sorted(
            episode_index for episode_index in selected if task_name in episode_tasks[episode_index]
        )
        if len(candidates) < trajectories_per_task:
            raise SystemExit(
                f"Task {suite_task_ids[task_name]} in {suite_name} has only {len(candidates)} "
                f"candidate trajectories, fewer than requested {trajectories_per_task}"
            )
        random.Random(selection_seed + dataset_task_ids[task_name]).shuffle(candidates)
        sampled.extend(candidates[:trajectories_per_task])
    selected = sorted(set(sampled))

selected.sort()
if not selected:
    raise SystemExit(f"No training episodes selected for {suite_name}")
print(json.dumps(selected, separators=(",", ":")))
PY
}

safe_tag() {
  local value="$1"
  value="${value//+/p}"
  value="${value//[^A-Za-z0-9_.-]/_}"
  echo "$value"
}

require_non_negative_int() {
  local name="$1"
  local value="${!name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer, got '${value}'." >&2
    exit 1
  fi
}

require_positive_number() {
  local name="$1"
  local value="${!name}"
  VALUE_NAME="$name" VALUE_TEXT="$value" "$PYTHON_BIN" - <<'PY'
import math
import os

name = os.environ["VALUE_NAME"]
text = os.environ["VALUE_TEXT"]
try:
    value = float(text)
except ValueError as exc:
    raise SystemExit(f"{name} must be a number, got {text!r}") from exc
if not math.isfinite(value) or value <= 0:
    raise SystemExit(f"{name} must be finite and positive, got {text!r}")
PY
}

run_group_for() {
  local suite="$1"
  local lambda_flow_k="$2"
  local phy_loss_weight="$3"
  local image_only_condition_jvp="$4"
  local interpolation_mode="$5"
  local dct_coe_num="$6"
  local bspline_degree="$7"
  local bspline_coe_num="$8"
  local interpolation_tag
  local image_only_tag=""
  local pretrain_tag=""
  if [[ "$image_only_condition_jvp" == "true" ]]; then
    image_only_tag="_image_only_jvptrue"
  fi
  if (( PRE_TRAIN_STEPS > 0 )); then
    pretrain_tag="_pre_train_steps$(safe_tag "$PRE_TRAIN_STEPS")"
  fi
  if [[ "$RUN_NAME_STYLE" == "kinematic_physical_steps_flow" ]]; then
    echo "${RUN_PREFIX}${TRAIN_TASK_TAG}_kinematic_$(safe_tag "$lambda_flow_k")_physical_$(safe_tag "$phy_loss_weight")_steps_$(safe_tag "$STEPS")_flow${image_only_tag}${pretrain_tag}"
    return
  fi
  if [[ "$RUN_NAME_STYLE" == "low_data_flowpolicy" ]]; then
    echo "${RUN_PREFIX}${TRAIN_TASK_TAG}_low_data_flowpolicy_kinematic_$(safe_tag "$lambda_flow_k")_physical_$(safe_tag "$phy_loss_weight")_steps_$(safe_tag "$STEPS")${image_only_tag}${pretrain_tag}"
    return
  fi
  if [[ "$RUN_NAME_STYLE" == "low_data_flowpolicy_bsp_num" ]]; then
    echo "${RUN_PREFIX}${TRAIN_TASK_TAG}_low_data_flowpolicy_kinematic_$(safe_tag "$lambda_flow_k")_physical_$(safe_tag "$phy_loss_weight")_steps_$(safe_tag "$STEPS")_bsp_num_$(safe_tag "$bspline_coe_num")${image_only_tag}${pretrain_tag}"
    return
  fi
  if [[ "$interpolation_mode" == "bspline" ]]; then
    interpolation_tag="bsp$(safe_tag "$bspline_degree")m$(safe_tag "$bspline_coe_num")"
  else
    interpolation_tag="dct$(safe_tag "$dct_coe_num")"
  fi
  # Keep the historical tag for the fixed action-JVP setting.
  echo "${RUN_PREFIX}_$(safe_tag "$suite")${TRAIN_TASK_TAG}_kinematic$(safe_tag "$lambda_flow_k")_phy$(safe_tag "$phy_loss_weight")_aktrue_tangent$(safe_tag "$POLICY_CONDITIONING_DERIVATIVE_MODE")${image_only_tag}_${interpolation_tag}_batch${BATCH_SIZE}_steps${STEPS}${pretrain_tag}"
}

run_path_for() {
  local kind="$1"
  local suite="$2"
  local seed="$3"
  local lambda_flow_k="$4"
  local phy_loss_weight="$5"
  local image_only_condition_jvp="$6"
  local interpolation_mode="$7"
  local dct_coe_num="$8"
  local bspline_degree="$9"
  local bspline_coe_num="${10}"
  local run_group
  run_group="$(run_group_for "$suite" "$lambda_flow_k" "$phy_loss_weight" "$image_only_condition_jvp" "$interpolation_mode" "$dct_coe_num" "$bspline_degree" "$bspline_coe_num")"
  if [[ "$APPEND_RUN_TIMESTAMP" == "true" ]]; then
    run_group="${run_group}_${RUN_TIMESTAMP}"
  fi
  if [[ "$kind" == "train_logs" && -n "$TRAIN_LOG_ROOT" ]]; then
    echo "${TRAIN_LOG_ROOT%/}/${run_group}/seed${seed}"
    return
  fi
  echo "${OUTPUT_ROOT%/}/${kind}/${run_group}/seed${seed}"
}

run_one() {
  local suite="$1"
  local suite_episodes="$2"
  local seed="$3"
  local gpu_group="$4"
  local lambda_flow_k="$5"
  local phy_loss_weight="$6"
  local image_only_condition_jvp="$7"
  local interpolation_mode="$8"
  local dct_coe_num="$9"
  local bspline_degree="${10}"
  local bspline_coe_num="${11}"
  local -a group_gpus
  local num_processes
  local out
  local eval_out
  local run_group
  local job_name
  local wandb_group
  local wandb_notes
  local wandb_tags

  IFS=',' read -r -a group_gpus <<< "$gpu_group"
  num_processes="${#group_gpus[@]}"

  out="$(run_path_for train "$suite" "$seed" "$lambda_flow_k" "$phy_loss_weight" "$image_only_condition_jvp" "$interpolation_mode" "$dct_coe_num" "$bspline_degree" "$bspline_coe_num")"
  eval_out="$(run_path_for eval "$suite" "$seed" "$lambda_flow_k" "$phy_loss_weight" "$image_only_condition_jvp" "$interpolation_mode" "$dct_coe_num" "$bspline_degree" "$bspline_coe_num")"
  run_group="$(run_group_for "$suite" "$lambda_flow_k" "$phy_loss_weight" "$image_only_condition_jvp" "$interpolation_mode" "$dct_coe_num" "$bspline_degree" "$bspline_coe_num")"
  job_name="${run_group}_seed${seed}"
  wandb_group="${WANDB_GROUP:-$run_group}"
  wandb_notes="${WANDB_NOTES:-LOG_DIR:${wandb_group}}"
  wandb_tags="${WANDB_TAGS:-[\"LOG_DIR:$(safe_tag "$wandb_group")\"]}"

  echo "suite=${suite} seed=${seed} gpu=${gpu_group} num_processes=${num_processes} lambda_flow_k=${lambda_flow_k} pre_train_steps=${PRE_TRAIN_STEPS} phy_loss_weight=${phy_loss_weight} tangent=${POLICY_CONDITIONING_DERIVATIVE_MODE} image_only_condition_jvp=${image_only_condition_jvp} interpolation_mode=${interpolation_mode} dct_coe_num=${dct_coe_num} bspline_degree=${bspline_degree} bspline_coe_num=${bspline_coe_num} out=${out} eval_out=${eval_out}"

  if [[ -d "$out" && "$RESUME" != "true" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "Output directory already exists: ${out}" >&2
    echo "Set RESUME=true or change RUN_PREFIX/OUTPUT_ROOT." >&2
    return 1
  fi

  local -a cmd=(
    "$ACCELERATE_BIN" launch
    --num_processes="${num_processes}"
    --mixed_precision="${MIXED_PRECISION}"
  )
  if (( num_processes > 1 )); then
    cmd+=(--multi_gpu --main_process_port=0)
  fi

  cmd+=(
    "$LEROBOT_TRAIN_BIN"
    --job_name="${job_name}"
    --resume="${RESUME}"
    --seed="${seed}"
    --wandb.enable="${WANDB_ENABLE}"
    --wandb.project="${WANDB_PROJECT}"
    --wandb.mode="${WANDB_MODE}"
    --wandb.group="${wandb_group}"
    --wandb.notes="${wandb_notes}"
    --wandb.tags="${wandb_tags}"
    --policy.type=flow
    --policy.device="${DEVICE}"
    --policy.use_amp=true
    --policy.n_obs_steps="${POLICY_N_OBS_STEPS}"
    --policy.horizon="${POLICY_HORIZON}"
    --policy.n_action_steps="${POLICY_N_ACTION_STEPS}"
    --policy.drop_n_last_frames="${POLICY_DROP_N_LAST_FRAMES}"
    --policy.vision_backbone="${POLICY_VISION_BACKBONE}"
    --policy.pretrained_backbone_weights="${POLICY_PRETRAINED_BACKBONE_WEIGHTS}"
    --policy.use_group_norm="${POLICY_USE_GROUP_NORM}"
    --policy.spatial_softmax_num_keypoints="${POLICY_SPATIAL_SOFTMAX_NUM_KEYPOINTS}"
    --policy.use_separate_rgb_encoder_per_camera="${POLICY_USE_SEPARATE_RGB_ENCODER_PER_CAMERA}"
    "--policy.resize_shape=${POLICY_RESIZE_SHAPE}"
    --policy.crop_ratio="${POLICY_CROP_RATIO}"
    --policy.crop_is_random="${POLICY_CROP_IS_RANDOM}"
    "--policy.down_dims=${POLICY_DOWN_DIMS}"
    --policy.kernel_size="${POLICY_KERNEL_SIZE}"
    --policy.n_groups="${POLICY_N_GROUPS}"
    --policy.diffusion_step_embed_dim="${POLICY_TIMESTEP_EMBED_DIM}"
    --policy.use_film_scale_modulation="${POLICY_USE_FILM_SCALE_MODULATION}"
    --policy.compile_model="${POLICY_COMPILE_MODEL}"
    --policy.num_inference_steps="${POLICY_NUM_INFERENCE_STEPS}"
    --policy.timestep_sampling_strategy="${POLICY_TIMESTEP_SAMPLING_STRATEGY}"
    --policy.timestep_sampling_alpha="${POLICY_TIMESTEP_SAMPLING_ALPHA}"
    --policy.timestep_sampling_beta="${POLICY_TIMESTEP_SAMPLING_BETA}"
    --policy.timestep_sampling_s="${POLICY_TIMESTEP_SAMPLING_S}"
    --policy.sigma_min="${POLICY_SIGMA_MIN}"
    --policy.lambda_flow_k="${lambda_flow_k}"
    --policy.pre_train_steps="${PRE_TRAIN_STEPS}"
    --policy.phy_loss_weight="${phy_loss_weight}"
    --policy.sample_frequency="${POLICY_SAMPLE_FREQUENCY}"
    --policy.gripper_first="${POLICY_GRIPPER_FIRST}"
    --policy.conditioning_derivative_mode="${POLICY_CONDITIONING_DERIVATIVE_MODE}"
    --policy.image_only_condition_jvp="${image_only_condition_jvp}"
    --policy.interpolation_mode="${interpolation_mode}"
    --policy.do_mask_loss_for_padding="${POLICY_DO_MASK_LOSS_FOR_PADDING}"
    --policy.optimizer_lr="${POLICY_OPTIMIZER_LR}"
    --policy.optimizer_weight_decay="${POLICY_OPTIMIZER_WEIGHT_DECAY}"
    --policy.scheduler_warmup_steps="${POLICY_SCHEDULER_WARMUP_STEPS}"
    --dataset.repo_id=HuggingFaceVLA/libero
    --dataset.root="${LIBERO_ROOT}"
    --dataset.episodes="${suite_episodes}"
    --dataset.image_transforms.enable=true
    --dataset.image_transforms.max_num_transforms=4
    '--dataset.image_transforms.tfs={"brightness":{"type":"ColorJitter","kwargs":{"brightness":[0.75,1.25]}},"contrast":{"type":"ColorJitter","kwargs":{"contrast":[0.6,1.4]}},"saturation":{"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}},"hue":{"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"type":"SharpnessJitter","kwargs":{"sharpness":[0.6,1.4]}},"rotation":{"type":"RandomRotation","kwargs":{"degrees":[-5,5]}},"translation":{"type":"RandomAffine","kwargs":{"degrees":0,"translate":[0.1,0.1]}}}'
    --dataset.video_backend=torchcodec
    --env_eval_freq="${ENV_EVAL_FREQ}"
    --env_eval_output_dir="${eval_out}"
    --env.type=libero
    --env.task="${suite}"
    --env.observation_height="${EVAL_OBSERVATION_HEIGHT}"
    --env.observation_width="${EVAL_OBSERVATION_WIDTH}"
    --eval.batch_size="${EVAL_BATCH_SIZE}"
    --eval.n_episodes="${EVAL_EPISODES}"
    --eval.use_async_envs="${EVAL_USE_ASYNC_ENVS}"
    --steps="${STEPS}"
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"
    --output_dir="${out}"
    --policy.push_to_hub=false
  )

  if [[ "$interpolation_mode" == "bspline" ]]; then
    cmd+=(
      --policy.bspline_degree="${bspline_degree}"
      --policy.bspline_coe_num="${bspline_coe_num}"
    )
  else
    cmd+=(--policy.dct_coe_num="${dct_coe_num}")
  fi

  if [[ -n "$TRAIN_TASK_IDS_JSON" ]]; then
    cmd+=(--env.task_ids="${TRAIN_TASK_IDS_JSON}")
  fi

  if [[ -n "$DECODED_IMAGE_CACHE_ROOT" ]]; then
    cmd+=(--dataset.decoded_image_cache_root="${DECODED_IMAGE_CACHE_ROOT}")
  fi
  if [[ -n "$WANDB_ENTITY" ]]; then
    cmd+=(--wandb.entity="${WANDB_ENTITY}")
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_group"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu_group}" "${cmd[@]}"
}

require_non_negative_int STEPS
require_non_negative_int BATCH_SIZE
require_non_negative_int NUM_WORKERS
require_non_negative_int SAVE_FREQ
require_non_negative_int LOG_FREQ
require_non_negative_int POLICY_N_OBS_STEPS
require_non_negative_int POLICY_HORIZON
require_non_negative_int POLICY_N_ACTION_STEPS
require_non_negative_int POLICY_DROP_N_LAST_FRAMES
require_non_negative_int POLICY_NUM_INFERENCE_STEPS
require_non_negative_int POLICY_SCHEDULER_WARMUP_STEPS
require_non_negative_int PRE_TRAIN_STEPS
require_non_negative_int ENV_EVAL_FREQ
require_non_negative_int EVAL_EPISODES
require_non_negative_int EVAL_BATCH_SIZE
require_non_negative_int EVAL_OBSERVATION_HEIGHT
require_non_negative_int EVAL_OBSERVATION_WIDTH
require_non_negative_int TRAJECTORY_SELECTION_SEED
require_positive_number POLICY_SAMPLE_FREQUENCY

if [[ -n "$TRAIN_TRAJECTORIES_PER_TASK" ]]; then
  require_non_negative_int TRAIN_TRAJECTORIES_PER_TASK
  if (( TRAIN_TRAJECTORIES_PER_TASK == 0 )); then
    echo "TRAIN_TRAJECTORIES_PER_TASK must be positive when supplied." >&2
    exit 1
  fi
fi

case "$RUN_NAME_STYLE" in
  full|kinematic_physical_steps_flow|low_data_flowpolicy|low_data_flowpolicy_bsp_num) ;;
  *)
    echo "Unsupported RUN_NAME_STYLE '${RUN_NAME_STYLE}'." >&2
    exit 1
    ;;
esac

case "$APPEND_RUN_TIMESTAMP" in
  true|false) ;;
  *)
    echo "APPEND_RUN_TIMESTAMP must be true or false; got '${APPEND_RUN_TIMESTAMP}'." >&2
    exit 1
    ;;
esac

if (( BATCH_SIZE == 0 || POLICY_N_OBS_STEPS == 0 || POLICY_HORIZON == 0 || POLICY_N_ACTION_STEPS == 0 || POLICY_NUM_INFERENCE_STEPS == 0 )); then
  echo "BATCH_SIZE and the policy observation/horizon/action/inference-step counts must be positive." >&2
  exit 1
fi
if (( POLICY_N_OBS_STEPS + POLICY_N_ACTION_STEPS - 1 > POLICY_HORIZON )); then
  echo "POLICY_N_OBS_STEPS + POLICY_N_ACTION_STEPS - 1 must not exceed POLICY_HORIZON." >&2
  exit 1
fi
if (( PRE_TRAIN_STEPS > STEPS )); then
  echo "PRE_TRAIN_STEPS must not exceed STEPS; got ${PRE_TRAIN_STEPS} > ${STEPS}." >&2
  exit 1
fi

case "${POLICY_CONDITIONING_DERIVATIVE_MODE,,}" in
  reverse|forward|central)
    POLICY_CONDITIONING_DERIVATIVE_MODE="${POLICY_CONDITIONING_DERIVATIVE_MODE,,}"
    ;;
  *)
    echo "POLICY_CONDITIONING_DERIVATIVE_MODE must be reverse, forward, or central; got '${POLICY_CONDITIONING_DERIVATIVE_MODE}'." >&2
    exit 1
    ;;
esac

POLICY_INTERPOLATION_MODE="${POLICY_INTERPOLATION_MODE,,}"
case "$POLICY_INTERPOLATION_MODE" in
  dct|bspline) ;;
  *)
    echo "INTERPOLATION_MODE/POLICY_INTERPOLATION_MODE must be dct or bspline; got '${POLICY_INTERPOLATION_MODE}'." >&2
    exit 1
    ;;
esac

mapfile -t SEEDS_ARRAY < <(parse_list "$RUN_SEEDS" seed)
mapfile -t GPUS_ARRAY < <(parse_list "$GPUS" gpu)
mapfile -t SUITES_ARRAY < <(parse_list "$TRAIN_SUITES" suite)
TRAIN_TASK_IDS_JSON=""
TRAIN_TASK_TAG=""
if [[ -n "$TRAIN_TASK_IDS" ]]; then
  mapfile -t TRAIN_TASK_IDS_ARRAY < <(parse_list "$TRAIN_TASK_IDS" task)
  if (( ${#TRAIN_TASK_IDS_ARRAY[@]} == 0 )); then
    echo "TRAIN_TASK_IDS must not be empty when supplied." >&2
    exit 1
  fi
  if (( $(printf '%s\n' "${TRAIN_TASK_IDS_ARRAY[@]}" | sort -nu | wc -l) != ${#TRAIN_TASK_IDS_ARRAY[@]} )); then
    echo "TRAIN_TASK_IDS contains duplicate ids: ${TRAIN_TASK_IDS}." >&2
    exit 1
  fi
  printf -v task_ids_joined ',%s' "${TRAIN_TASK_IDS_ARRAY[@]}"
  TRAIN_TASK_IDS_JSON="[${task_ids_joined:1}]"
  printf -v task_tag_joined -- '-%s' "${TRAIN_TASK_IDS_ARRAY[@]}"
  if (( ${#TRAIN_TASK_IDS_ARRAY[@]} == 1 )); then
    TRAIN_TASK_TAG="_task${TRAIN_TASK_IDS_ARRAY[0]}"
  else
    TRAIN_TASK_TAG="_tasks${task_tag_joined:1}"
  fi
fi
if [[ -n "$TRAIN_TRAJECTORIES_PER_TASK" ]]; then
  TRAIN_TASK_TAG="${TRAIN_TASK_TAG}_traj${TRAIN_TRAJECTORIES_PER_TASK}seed${TRAJECTORY_SELECTION_SEED}"
fi
mapfile -t LAMBDAS < <(parse_list "$POLICY_LAMBDA_FLOW_K" lambda_flow_k)
mapfile -t PHY_LOSS_WEIGHTS < <(parse_list "$PHY_LOSS_WEIGHT" phy_loss_weight)
mapfile -t IMAGE_ONLY_CONDITION_JVPS < <(parse_list "$POLICY_IMAGE_ONLY_CONDITION_JVP" image_only_condition_jvp)
mapfile -t GRIPPER_FIRST_VALUES < <(parse_list "$POLICY_GRIPPER_FIRST" gripper_first)
if (( ${#GRIPPER_FIRST_VALUES[@]} != 1 )); then
  echo "POLICY_GRIPPER_FIRST must contain exactly one boolean value." >&2
  exit 1
fi
POLICY_GRIPPER_FIRST="${GRIPPER_FIRST_VALUES[0]}"

INTERPOLATION_SPECS=()
DCT_COE_NUMS=()
BSPLINE_DEGREES=()
BSPLINE_COE_NUMS=()
if [[ "$POLICY_INTERPOLATION_MODE" == "dct" ]]; then
  mapfile -t DCT_COE_NUMS < <(parse_list "$POLICY_DCT_COE_NUM" dct_coe_num)
  for dct_coe_num in "${DCT_COE_NUMS[@]}"; do
    INTERPOLATION_SPECS+=("dct|${dct_coe_num}||")
  done
else
  mapfile -t BSPLINE_DEGREES < <(parse_list "$POLICY_BSPLINE_DEGREE" bspline_degree)
  mapfile -t BSPLINE_COE_NUMS < <(parse_list "$POLICY_BSPLINE_COE_NUM" bspline_coe_num)
  for bspline_degree in "${BSPLINE_DEGREES[@]}"; do
    for bspline_coe_num in "${BSPLINE_COE_NUMS[@]}"; do
      if (( bspline_degree >= bspline_coe_num )); then
        echo "B-spline mode requires degree < coefficient count, got p=${bspline_degree}, M=${bspline_coe_num}." >&2
        exit 1
      fi
      INTERPOLATION_SPECS+=("bspline||${bspline_degree}|${bspline_coe_num}")
    done
  done
fi

if (( ${#SEEDS_ARRAY[@]} == 0 || ${#GPUS_ARRAY[@]} == 0 || ${#SUITES_ARRAY[@]} == 0 || ${#LAMBDAS[@]} == 0 || ${#PHY_LOSS_WEIGHTS[@]} == 0 || ${#IMAGE_ONLY_CONDITION_JVPS[@]} == 0 || ${#INTERPOLATION_SPECS[@]} == 0 )); then
  echo "Empty suite/seed/GPU/lambda/phy_loss_weight/image_only_condition_jvp/interpolation parameter list." >&2
  exit 1
fi

declare -A SUITE_EPISODES
declare -A SUITE_EPISODE_COUNTS
for suite in "${SUITES_ARRAY[@]}"; do
  suite_episodes="$(resolve_suite_episodes "$suite")"
  SUITE_EPISODES["$suite"]="$suite_episodes"
  episode_body="${suite_episodes#[}"
  episode_body="${episode_body%]}"
  IFS=',' read -r -a episode_ids <<< "$episode_body"
  SUITE_EPISODE_COUNTS["$suite"]="${#episode_ids[@]}"
done

RUNS=()
for suite in "${SUITES_ARRAY[@]}"; do
  for seed in "${SEEDS_ARRAY[@]}"; do
    for lambda_flow_k in "${LAMBDAS[@]}"; do
      for phy_loss_weight in "${PHY_LOSS_WEIGHTS[@]}"; do
        for image_only_condition_jvp in "${IMAGE_ONLY_CONDITION_JVPS[@]}"; do
          for interpolation_spec in "${INTERPOLATION_SPECS[@]}"; do
            RUNS+=("${suite}|${seed}|${lambda_flow_k}|${phy_loss_weight}|${image_only_condition_jvp}|${interpolation_spec}")
          done
        done
      done
    done
  done
done

NUM_RUNS="${#RUNS[@]}"
NUM_GPU_GROUPS="${#GPUS_ARRAY[@]}"

echo "seeds=${SEEDS_ARRAY[*]}"
echo "gpus=${GPUS_ARRAY[*]}"
echo "train_suites=${SUITES_ARRAY[*]}"
echo "train_task_ids=${TRAIN_TASK_IDS_JSON:-all}"
echo "train_trajectories_per_task=${TRAIN_TRAJECTORIES_PER_TASK:-all}"
if [[ -n "$TRAIN_TRAJECTORIES_PER_TASK" ]]; then
  echo "trajectory_selection_seed=${TRAJECTORY_SELECTION_SEED}"
fi
for suite in "${SUITES_ARRAY[@]}"; do
  echo "suite=${suite} training_episodes=${SUITE_EPISODE_COUNTS[$suite]}"
done
echo "batch_size=${BATCH_SIZE}"
echo "steps=${STEPS}"
echo "env_eval_freq=${ENV_EVAL_FREQ}"
echo "eval_episodes=${EVAL_EPISODES}"
echo "decoded_image_cache_root=${DECODED_IMAGE_CACHE_ROOT:-disabled}"
echo "lambda_flow_k=${LAMBDAS[*]}"
echo "pre_train_steps=${PRE_TRAIN_STEPS}"
echo "phy_loss_weight=${PHY_LOSS_WEIGHTS[*]}"
echo "image_only_condition_jvp=${IMAGE_ONLY_CONDITION_JVPS[*]}"
echo "gripper_first=${POLICY_GRIPPER_FIRST}"
echo "sample_frequency=${POLICY_SAMPLE_FREQUENCY}"
echo "conditioning_derivative_mode=${POLICY_CONDITIONING_DERIVATIVE_MODE}"
echo "interpolation_mode=${POLICY_INTERPOLATION_MODE}"
if [[ "$POLICY_INTERPOLATION_MODE" == "dct" ]]; then
  echo "dct_coe_num=${DCT_COE_NUMS[*]}"
else
  echo "bspline_degree=${BSPLINE_DEGREES[*]}"
  echo "bspline_coe_num=${BSPLINE_COE_NUMS[*]}"
fi
echo "output_root=${OUTPUT_ROOT}"
echo "train_log_root=${TRAIN_LOG_ROOT:-${OUTPUT_ROOT%/}/train_logs}"
echo "run_name_style=${RUN_NAME_STYLE}"
echo "append_run_timestamp=${APPEND_RUN_TIMESTAMP}"
echo "run_timestamp=${RUN_TIMESTAMP}"
echo "runs=${NUM_RUNS}"

status=0
for ((start = 0; start < NUM_RUNS; start += NUM_GPU_GROUPS)); do
  pids=()
  labels=()
  logs=()

  for ((gpu_group_idx = 0; gpu_group_idx < NUM_GPU_GROUPS; gpu_group_idx++)); do
    run_idx=$((start + gpu_group_idx))
    if (( run_idx >= NUM_RUNS )); then
      break
    fi

    IFS='|' read -r suite seed lambda_flow_k phy_loss_weight image_only_condition_jvp interpolation_mode dct_coe_num bspline_degree bspline_coe_num <<< "${RUNS[$run_idx]}"
    suite_episodes="${SUITE_EPISODES[$suite]}"
    gpu_group="${GPUS_ARRAY[$gpu_group_idx]}"
    log_file="$(run_path_for train_logs "$suite" "$seed" "$lambda_flow_k" "$phy_loss_weight" "$image_only_condition_jvp" "$interpolation_mode" "$dct_coe_num" "$bspline_degree" "$bspline_coe_num").log"
    label="suite ${suite}, seed ${seed}, lambda_flow_k ${lambda_flow_k}, phy_loss_weight ${phy_loss_weight}, image_only_condition_jvp ${image_only_condition_jvp}, interpolation_mode ${interpolation_mode}, dct_coe_num ${dct_coe_num}, bspline_degree ${bspline_degree}, bspline_coe_num ${bspline_coe_num}, gpus ${gpu_group}"

    mkdir -p "$(dirname "$log_file")"
    echo "launch ${label}; log=${log_file}"
    (run_one "$suite" "$suite_episodes" "$seed" "$gpu_group" "$lambda_flow_k" "$phy_loss_weight" "$image_only_condition_jvp" "$interpolation_mode" "$dct_coe_num" "$bspline_degree" "$bspline_coe_num") >"$log_file" 2>&1 &
    pids+=("$!")
    labels+=("$label")
    logs+=("$log_file")
  done

  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "done ${labels[$idx]}; log=${logs[$idx]}"
    else
      echo "failed ${labels[$idx]}; log=${logs[$idx]}" >&2
      status=1
    fi
  done
done

exit "$status"
