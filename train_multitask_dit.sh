#!/usr/bin/env bash
# MultiTaskDiT flow-matching sweep for all 40 LIBERO tasks with the official CLIP architecture.
#

  # GPUS='["0,1,2,3","4,5,6,7"]'
  # RUN_SEEDS='[1000,1001]'
  # 对应关系是：
  #  实验         使用 GPU
  # ━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━
  #  seed 1000    GPU 0、1、2、3 共同训练
  # ───────────  ─────────────────────────
  #  seed 1001    GPU 4、5、6、7 共同训练
  # 两个实验并行运行；每个实验内部用四卡 DDP，同一个模型、不同数据 batch、同步梯度。
  # 如果有四个实验：
  # RUN_SEEDS='[1000,1001,1002,1003]'
  # 则分两批：
  # 1. seed 1000 → 0–3，seed 1001 → 4–7
  # 2. 两者都结束后，seed 1002 → 0–3，seed 1003 → 4–7
  # 如果只有一个 GPU 组：
  # GPUS='["0,1,2,3"]'
  # 那么所有实验会在这一个四卡组上依次运行，不会同时挤在该组上。


  # # 两个实验并行，每个实验使用四张卡
  # GPUS='["0,1,2,3"]' \
  # RUN_SEEDS='[1000]' \
  # BATCH_SIZE=80 \
  # STEPS=100000 \
  # NUM_WORKERS=20 \
  # POLICY_LAMBDA_FLOW_K='[0.01]' \
  # PHY_LOSS_WEIGHT='[0.0,0.01,0.1]' \
  # INTERPOLATION_MODE=dct \
  # POLICY_DCT_COE_NUM='[48]' \
  # ENV_EVAL_FREQ=0 \
  # bash train_multitask_dit.sh

  # # B-spline starting point (not a claimed optimum): p and M are required;
  # # M < H smooths noise and should be tuned on the target robot data.
  # INTERPOLATION_MODE=bspline \
  # BSPLINE_DEGREE='[3]' \
  # BSPLINE_COE_NUM='[24]' \
  # RUN_SEEDS='[1000]' \
  # bash train_multitask_dit.sh

  # # 两个独立单卡实验
  # GPUS='0,1' RUN_SEEDS='[1000,1001]' bash train_multitask_dit.sh

  # GPUS='1' \
  # RUN_SEEDS='[1000]' \
  # BATCH_SIZE=64 \
  # STEPS=20000 \
  # NUM_WORKERS=16 \
  # POLICY_LAMBDA_FLOW_K='[0.01]' \
  # PHY_LOSS_WEIGHT=0.0 \
  # POLICY_DCT_COE_NUM='[40]' \
  # ENV_EVAL_FREQ=20000 \
  # bash train_multitask_dit.sh


  # - 单卡实验，NUM_WORKERS=24 → 总共 24 个
  # - 四卡实验，NUM_WORKERS=24 → 总共 96 个
  # - 两个四卡实验，NUM_WORKERS=24 → 总共 192 个


# # 原 DCT
# INTERPOLATION_MODE=dct \
# POLICY_DCT_COE_NUM='[48]' \
# bash train_multitask_dit.sh

# # B-spline；p/M 均支持列表 sweep
# INTERPOLATION_MODE=bspline \
# BSPLINE_DEGREE='[3]' \
# BSPLINE_COE_NUM='[24,32,48]' \
# bash train_multitask_dit.sh


# evaluation
# batch_size=10：每个 task 创建 10 个 EGL worker。
# batch_size=50：每个 task 创建 50 个。


# 显存大是因为现在对两帧都求了jvp
# 补两帧都加的结果
set -euo pipefail

cd "$(dirname "$0")"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ "$MUJOCO_GL" == "egl" ]]; then
  export PYOPENGL_PLATFORM="egl"

  # libnvidia-gl normally installs this GLVND registration system-wide.  Some
  # machines in this cluster are missing it, which makes EGL fall back to Mesa
  # and fail while opening /dev/dri.  Keep a project-local NVIDIA registration
  # as a no-root fallback.
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

LIBERO_ROOT="${LIBERO_ROOT:-/home/zhouzhi/.cache/huggingface/lerobot/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e}"
# Optional explicit episode subset, expressed as a JSON/Python list.  Keeping
# this at the dataset-loader level avoids copying the multi-gigabyte dataset
# and guarantees every compared objective sees exactly the same trajectories.
DATASET_EPISODES="${DATASET_EPISODES:-}"
# Use the lossless uint8 mmap cache automatically only after its manifest is
# complete. Explicitly set DECODED_IMAGE_CACHE_ROOT="" to force the historical
# Parquet/PIL path for A/B checks.
DEFAULT_DECODED_IMAGE_CACHE_ROOT="/home/zhouzhi/.cache/lerobot_decoded/HuggingFaceVLA--libero/86958911c0f959db2bbbdb107eb3e17c5f9c798e"
if [[ -v DECODED_IMAGE_CACHE_ROOT ]]; then
  DECODED_IMAGE_CACHE_ROOT="${DECODED_IMAGE_CACHE_ROOT}"
elif [[ -f "${DEFAULT_DECODED_IMAGE_CACHE_ROOT}/manifest.json" ]]; then
  DECODED_IMAGE_CACHE_ROOT="${DEFAULT_DECODED_IMAGE_CACHE_ROOT}"
else
  DECODED_IMAGE_CACHE_ROOT=""
fi

# One training run uses these three GPUs together unless overridden.
DEFAULT_GPU_GROUPS='["0,2,4"]'
GPUS="${GPUS:-${GPU_IDS:-${GPU_ID:-${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPU_GROUPS}}}}"


PRE_TRAIN_STEPS="${PRE_TRAIN_STEPS:-${POLICY_PRE_TRAIN_STEPS:-0}}"

# POLICY_LAMBDA_FLOW_K="${POLICY_LAMBDA_FLOW_K:-0.0}"
# PHY_LOSS_WEIGHT="${PHY_LOSS_WEIGHT:-0.0}"
# POLICY_DCT_COE_NUM="${POLICY_DCT_COE_NUM:-0}"
POLICY_LAMBDA_FLOW_K="${POLICY_LAMBDA_FLOW_K:-0.01 0.1 1}"
PHY_LOSS_WEIGHT="${PHY_LOSS_WEIGHT:-0}"
# Kinematic supervision always includes the action JVP with gradients enabled.

STEPS="${STEPS:-20000}"
POLICY_INTERPOLATION_MODE="${POLICY_INTERPOLATION_MODE:-${INTERPOLATION_MODE:-bspline}}"
POLICY_DCT_COE_NUM="${POLICY_DCT_COE_NUM:-48}"
POLICY_BSPLINE_DEGREE="${POLICY_BSPLINE_DEGREE:-${BSPLINE_DEGREE:-2}}"
POLICY_BSPLINE_COE_NUM="${POLICY_BSPLINE_COE_NUM:-${BSPLINE_COE_NUM:-48}}"
# Forward differences for conditioning velocity: (s[t+1] - s[t]) / dt.
POLICY_CONDITIONING_DERIVATIVE_MODE="${POLICY_CONDITIONING_DERIVATIVE_MODE:-forward}"
POLICY_IMAGE_ONLY_CONDITION_JVP="${POLICY_IMAGE_ONLY_CONDITION_JVP:-false}"
POLICY_NUM_INTEGRATION_STEPS="${POLICY_NUM_INTEGRATION_STEPS:-32}"

RUN_SEEDS="${RUN_SEEDS:-${SEEDS:-${SEED:-1000}}}"
BATCH_SIZE="${BATCH_SIZE:-100}"
# Effective batch = BATCH_SIZE * GPUs per run * GRADIENT_ACCUMULATION_STEPS.
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-12}"
# check point
SAVE_FREQ="${SAVE_FREQ:-10000}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-false}"

ENV_EVAL_FREQ="${ENV_EVAL_FREQ:-20000}"
EVAL_SUITES="${EVAL_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
EVAL_USE_ASYNC_ENVS="${EVAL_USE_ASYNC_ENVS:-true}"
EVAL_MAX_EPISODES_RENDERED="${EVAL_MAX_EPISODES_RENDERED:-0}"
EVAL_OBSERVATION_HEIGHT="${EVAL_OBSERVATION_HEIGHT:-256}"
EVAL_OBSERVATION_WIDTH="${EVAL_OBSERVATION_WIDTH:-256}"

RUN_PREFIX="${RUN_PREFIX:-multitask_dit_flow}"
# RUN_PREFIX="${RUN_PREFIX:-multitask-dit-flow-libero-baseline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-}"
TRAIN_LOG_INCLUDE_EVAL_EPISODES="${TRAIN_LOG_INCLUDE_EVAL_EPISODES:-false}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
# Opt-in compact output names for targeted follow-up runs. The default keeps
# every historical path byte-for-byte unchanged.
RUN_NAME_STYLE="${RUN_NAME_STYLE:-full}"
APPEND_RUN_TIMESTAMP="${APPEND_RUN_TIMESTAMP:-true}"

WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-multitask_dit_libero}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-${WANDB_RUN_GROUP:-}}"
WANDB_NOTES="${WANDB_NOTES:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

parse_list() {
  local spec="$1"
  local kind="$2"
  LIST_SPEC="$spec" LIST_KIND="$kind" "$PYTHON_BIN" - <<'PY'
import ast
import math
import os
import re

spec = os.environ["LIST_SPEC"].strip()
kind = os.environ["LIST_KIND"]

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
        if kind == "dct_coe_num" and value > 48:
            raise SystemExit(f"dct_coe_num must not exceed the configured horizon 48, got {value}")
        if kind == "bspline_degree" and not 2 <= value < 48:
            raise SystemExit(f"bspline_degree must be in [2, 47] for horizon 48, got {value}")
        if kind == "bspline_coe_num" and not 3 <= value <= 48:
            raise SystemExit(f"bspline_coe_num must be in [3, 48] for horizon 48, got {value}")
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

for value in values:
    if kind == "seed":
        value = int(value)
        if value < 0:
            raise SystemExit(f"{kind} must be non-negative")
        print(value)
    elif kind in {"lambda_flow_k", "phy_loss_weight"}:
        print(value)
    elif kind in {"dct_coe_num", "bspline_degree", "bspline_coe_num"}:
        print(value)
    elif kind == "image_only_condition_jvp":
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

run_group_for() {
  local lambda_flow_k="$1"
  local phy_loss_weight="$2"
  local image_only_condition_jvp="$3"
  local interpolation_mode="$4"
  local dct_coe_num="$5"
  local bspline_degree="$6"
  local bspline_coe_num="$7"
  local interpolation_tag
  local accumulation_tag=""
  if (( GRADIENT_ACCUMULATION_STEPS > 1 )); then
    accumulation_tag="_ga${GRADIENT_ACCUMULATION_STEPS}"
  fi
  if [[ "$RUN_NAME_STYLE" == "kinematic_physical_steps" ]]; then
    echo "${RUN_PREFIX}_kinematic_$(safe_tag "$lambda_flow_k")_physical_$(safe_tag "$phy_loss_weight")_steps_$(safe_tag "$STEPS")${accumulation_tag}"
    return
  fi
  if [[ "$interpolation_mode" == "bspline" ]]; then
    # Compact because the legacy RUN_PREFIX already sits close to Linux's
    # 255-byte filename-component limit.
    interpolation_tag="bsp$(safe_tag "$bspline_degree")m$(safe_tag "$bspline_coe_num")"
  else
    # Keep the historical DCT run name byte-for-byte compatible.
    interpolation_tag="dct_$(safe_tag "$dct_coe_num")"
  fi
  # Keep the historical tags for the fixed action-JVP settings.
  echo "${RUN_PREFIX}_kinematic_$(safe_tag "$lambda_flow_k")_phy_$(safe_tag "$phy_loss_weight")_use_ak_true_stopgrad_ak_false_tangent_$(safe_tag "$POLICY_CONDITIONING_DERIVATIVE_MODE")_image_only_jvp_$(safe_tag "$image_only_condition_jvp")_${interpolation_tag}_batch_size${BATCH_SIZE}_steps${STEPS}_pre_train_steps${PRE_TRAIN_STEPS}${accumulation_tag}"
}

run_path_for() {
  local kind="$1"
  local seed="$2"
  local lambda_flow_k="$3"
  local phy_loss_weight="$4"
  local image_only_condition_jvp="$5"
  local interpolation_mode="$6"
  local dct_coe_num="$7"
  local bspline_degree="$8"
  local bspline_coe_num="$9"
  local run_group
  run_group="$(run_group_for "$lambda_flow_k" "$phy_loss_weight" "$image_only_condition_jvp" "$interpolation_mode" "$dct_coe_num" "$bspline_degree" "$bspline_coe_num")"
  if [[ "$APPEND_RUN_TIMESTAMP" == "true" ]]; then
    run_group="${run_group}_${RUN_TIMESTAMP}"
  fi
  if [[ "$kind" == "train_logs" && -n "$TRAIN_LOG_ROOT" ]]; then
    local log_stem="seed${seed}"
    if [[ "$TRAIN_LOG_INCLUDE_EVAL_EPISODES" == "true" ]]; then
      log_stem="${log_stem}_eval_episodes${EVAL_EPISODES}"
    fi
    echo "${TRAIN_LOG_ROOT%/}/${run_group}/${log_stem}"
    return
  fi
  echo "${OUTPUT_ROOT%/}/${kind}/${run_group}/seed${seed}"
}

run_one() {
  local seed="$1"
  local gpu_group="$2"
  local lambda_flow_k="$3"
  local phy_loss_weight="$4"
  local image_only_condition_jvp="$5"
  local interpolation_mode="$6"
  local dct_coe_num="$7"
  local bspline_degree="$8"
  local bspline_coe_num="$9"
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

  out="$(run_path_for train "$seed" "$lambda_flow_k" "$phy_loss_weight" "$image_only_condition_jvp" "$interpolation_mode" "$dct_coe_num" "$bspline_degree" "$bspline_coe_num")"
  eval_out="$(run_path_for eval "$seed" "$lambda_flow_k" "$phy_loss_weight" "$image_only_condition_jvp" "$interpolation_mode" "$dct_coe_num" "$bspline_degree" "$bspline_coe_num")"
  run_group="$(run_group_for "$lambda_flow_k" "$phy_loss_weight" "$image_only_condition_jvp" "$interpolation_mode" "$dct_coe_num" "$bspline_degree" "$bspline_coe_num")"
  job_name="${run_group}_seed${seed}"
  wandb_group="${WANDB_GROUP:-$run_group}"
  wandb_notes="${WANDB_NOTES:-LOG_DIR:${wandb_group}}"
  wandb_tags="${WANDB_TAGS:-[\"LOG_DIR:$(safe_tag "$wandb_group")\"]}"

  echo "seed=${seed} gpu=${gpu_group} num_processes=${num_processes} eval_suites=${EVAL_SUITES} lambda_flow_k=${lambda_flow_k} phy_loss_weight=${phy_loss_weight} tangent=${POLICY_CONDITIONING_DERIVATIVE_MODE} image_only_condition_jvp=${image_only_condition_jvp} interpolation_mode=${interpolation_mode} dct_coe_num=${dct_coe_num} bspline_degree=${bspline_degree} bspline_coe_num=${bspline_coe_num} pre_train_steps=${PRE_TRAIN_STEPS} out=${out} eval_out=${eval_out} wandb_group=${wandb_group}"

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
    --policy.type=multi_task_dit
    --policy.device="${DEVICE}"
    --policy.use_amp=true
    --policy.single_task=false
    --policy.objective=flow_matching
    --policy.n_obs_steps=2
    --policy.horizon=48
    --policy.n_action_steps=24
    --policy.drop_n_last_frames=23
    --policy.use_rope=true
    --policy.use_positional_encoding=false
    --policy.hidden_dim=768
    --policy.num_layers=8
    --policy.num_heads=12
    --policy.dropout=0.1
    --policy.timestep_embed_dim=256
    --policy.optimizer_lr=3e-4
    --policy.optimizer_weight_decay=0
    --policy.scheduler_warmup_steps=0
    --policy.vision_encoder_lr_multiplier=0.1
    --policy.vision_encoder_type=clip
    --policy.vision_encoder_name=openai/clip-vit-base-patch16
    --policy.vision_backbone=resnet18
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
    --policy.spatial_softmax_num_keypoints=32
    --policy.use_separate_rgb_encoder_per_camera=false
    "--policy.image_resize_shape=[256,256]"
    "--policy.image_crop_shape=[224,224]"
    --policy.image_crop_is_random=true
    --policy.text_encoder_name=openai/clip-vit-base-patch16
    --policy.do_mask_loss_for_padding=false
    --policy.sigma_min=0.0
    --policy.lambda_flow_k="${lambda_flow_k}"
    --policy.phy_loss_weight="${phy_loss_weight}"
    --policy.pre_train_steps="${PRE_TRAIN_STEPS}"
    --policy.conditioning_derivative_mode="${POLICY_CONDITIONING_DERIVATIVE_MODE}"
    --policy.image_only_condition_jvp="${image_only_condition_jvp}"
    --policy.gripper_first=false
    --policy.sample_frequency=10.0
    --policy.interpolation_mode="${interpolation_mode}"
    --policy.num_integration_steps="${POLICY_NUM_INTEGRATION_STEPS}"
    --policy.integration_method=euler
    --policy.timestep_sampling_strategy=beta
    --policy.timestep_sampling_alpha=1.5
    --policy.timestep_sampling_beta=1.0
    --policy.timestep_sampling_s=0.999
    --dataset.repo_id=HuggingFaceVLA/libero
    --dataset.root="${LIBERO_ROOT}"
    --dataset.image_transforms.enable=true
    --dataset.image_transforms.max_num_transforms=4
    '--dataset.image_transforms.tfs={"brightness":{"type":"ColorJitter","kwargs":{"brightness":[0.75,1.25]}},"contrast":{"type":"ColorJitter","kwargs":{"contrast":[0.6,1.4]}},"saturation":{"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}},"hue":{"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"type":"SharpnessJitter","kwargs":{"sharpness":[0.6,1.4]}},"rotation":{"type":"RandomRotation","kwargs":{"degrees":[-5,5]}},"translation":{"type":"RandomAffine","kwargs":{"degrees":0,"translate":[0.1,0.1]}}}'
    --dataset.video_backend=torchcodec
    --env_eval_freq="${ENV_EVAL_FREQ}"
    --env_eval_output_dir="${eval_out}"
    --env.type=libero
    --env.task="${EVAL_SUITES}"
    --env.observation_height="${EVAL_OBSERVATION_HEIGHT}"
    --env.observation_width="${EVAL_OBSERVATION_WIDTH}"
    --eval.batch_size="${EVAL_BATCH_SIZE}"
    --eval.n_episodes="${EVAL_EPISODES}"
    --eval.use_async_envs="${EVAL_USE_ASYNC_ENVS}"
    --eval.max_episodes_rendered="${EVAL_MAX_EPISODES_RENDERED}"
    --steps="${STEPS}"
    --batch_size="${BATCH_SIZE}"
    --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
    --num_workers="${NUM_WORKERS}"
    --save_freq="${SAVE_FREQ}"
    # Raw and weighted physical losses are logged separately on every step.
    --log_freq=100
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

  if [[ -n "$DATASET_EPISODES" ]]; then
    cmd+=(--dataset.episodes="${DATASET_EPISODES}")
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
require_non_negative_int GRADIENT_ACCUMULATION_STEPS
if (( GRADIENT_ACCUMULATION_STEPS == 0 )); then
  echo "GRADIENT_ACCUMULATION_STEPS must be positive." >&2
  exit 1
fi
require_non_negative_int PRE_TRAIN_STEPS
require_non_negative_int SAVE_FREQ
require_non_negative_int ENV_EVAL_FREQ
require_non_negative_int EVAL_EPISODES
require_non_negative_int EVAL_BATCH_SIZE
require_non_negative_int EVAL_OBSERVATION_HEIGHT
require_non_negative_int EVAL_OBSERVATION_WIDTH
require_non_negative_int POLICY_NUM_INTEGRATION_STEPS
if (( POLICY_NUM_INTEGRATION_STEPS == 0 )); then
  echo "POLICY_NUM_INTEGRATION_STEPS must be positive." >&2
  exit 1
fi

case "$RUN_NAME_STYLE" in
  full|kinematic_physical_steps) ;;
  *)
    echo "RUN_NAME_STYLE must be full or kinematic_physical_steps; got '${RUN_NAME_STYLE}'." >&2
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

case "$TRAIN_LOG_INCLUDE_EVAL_EPISODES" in
  true|false) ;;
  *)
    echo "TRAIN_LOG_INCLUDE_EVAL_EPISODES must be true or false; got '${TRAIN_LOG_INCLUDE_EVAL_EPISODES}'." >&2
    exit 1
    ;;
esac

case "$POLICY_CONDITIONING_DERIVATIVE_MODE" in
  reverse|forward|central) ;;
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
mapfile -t LAMBDAS < <(parse_list "$POLICY_LAMBDA_FLOW_K" lambda_flow_k)
mapfile -t PHY_LOSS_WEIGHTS < <(parse_list "$PHY_LOSS_WEIGHT" phy_loss_weight)
mapfile -t IMAGE_ONLY_CONDITION_JVPS < <(parse_list "$POLICY_IMAGE_ONLY_CONDITION_JVP" image_only_condition_jvp)

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
  if [[ -z "$POLICY_BSPLINE_DEGREE" || -z "$POLICY_BSPLINE_COE_NUM" ]]; then
    echo "bspline mode requires explicit BSPLINE_DEGREE/POLICY_BSPLINE_DEGREE and BSPLINE_COE_NUM/POLICY_BSPLINE_COE_NUM." >&2
    exit 1
  fi
  mapfile -t BSPLINE_DEGREES < <(parse_list "$POLICY_BSPLINE_DEGREE" bspline_degree)
  mapfile -t BSPLINE_COE_NUMS < <(parse_list "$POLICY_BSPLINE_COE_NUM" bspline_coe_num)
  for bspline_degree in "${BSPLINE_DEGREES[@]}"; do
    for bspline_coe_num in "${BSPLINE_COE_NUMS[@]}"; do
      if (( bspline_degree >= bspline_coe_num )); then
        echo "bspline mode requires p < M, got p=${bspline_degree}, M=${bspline_coe_num}." >&2
        exit 1
      fi
      INTERPOLATION_SPECS+=("bspline||${bspline_degree}|${bspline_coe_num}")
    done
  done
fi

if (( ${#SEEDS_ARRAY[@]} == 0 || ${#GPUS_ARRAY[@]} == 0 || ${#LAMBDAS[@]} == 0 || ${#PHY_LOSS_WEIGHTS[@]} == 0 || ${#IMAGE_ONLY_CONDITION_JVPS[@]} == 0 || ${#INTERPOLATION_SPECS[@]} == 0 )); then
  echo "Empty seed/GPU/lambda/phy_loss_weight/image_only_condition_jvp/interpolation parameter list." >&2
  exit 1
fi

RUNS=()
for seed in "${SEEDS_ARRAY[@]}"; do
  for lambda_flow_k in "${LAMBDAS[@]}"; do
    for phy_loss_weight in "${PHY_LOSS_WEIGHTS[@]}"; do
      for image_only_condition_jvp in "${IMAGE_ONLY_CONDITION_JVPS[@]}"; do
        for interpolation_spec in "${INTERPOLATION_SPECS[@]}"; do
          RUNS+=("${seed}|${lambda_flow_k}|${phy_loss_weight}|${image_only_condition_jvp}|${interpolation_spec}")
        done
      done
    done
  done
done

NUM_RUNS="${#RUNS[@]}"
NUM_GPU_GROUPS="${#GPUS_ARRAY[@]}"

echo "seeds=${SEEDS_ARRAY[*]}"
echo "gpus=${GPUS_ARRAY[*]}"
echo "batch_size=${BATCH_SIZE}"
echo "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
echo "steps=${STEPS}"
echo "pre_train_steps=${PRE_TRAIN_STEPS}"
echo "env_eval_freq=${ENV_EVAL_FREQ}"
echo "eval_suites=${EVAL_SUITES}"
echo "eval_episodes=${EVAL_EPISODES}"
echo "eval_batch_size=${EVAL_BATCH_SIZE}"
echo "eval_max_episodes_rendered=${EVAL_MAX_EPISODES_RENDERED}"
echo "decoded_image_cache_root=${DECODED_IMAGE_CACHE_ROOT:-disabled}"
echo "lambda_flow_k=${LAMBDAS[*]}"
echo "phy_loss_weight=${PHY_LOSS_WEIGHTS[*]}"
echo "conditioning_derivative_mode=${POLICY_CONDITIONING_DERIVATIVE_MODE}"
echo "image_only_condition_jvp=${IMAGE_ONLY_CONDITION_JVPS[*]}"
echo "num_integration_steps=${POLICY_NUM_INTEGRATION_STEPS}"
echo "interpolation_mode=${POLICY_INTERPOLATION_MODE}"
if [[ "$POLICY_INTERPOLATION_MODE" == "dct" ]]; then
  echo "dct_coe_num=${DCT_COE_NUMS[*]}"
else
  echo "bspline_degree=${BSPLINE_DEGREES[*]}"
  echo "bspline_coe_num=${BSPLINE_COE_NUMS[*]}"
fi
echo "output_root=${OUTPUT_ROOT}"
echo "train_log_root=${TRAIN_LOG_ROOT:-${OUTPUT_ROOT%/}/train_logs}"
echo "train_log_include_eval_episodes=${TRAIN_LOG_INCLUDE_EVAL_EPISODES}"
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

    IFS='|' read -r seed lambda_flow_k phy_loss_weight image_only_condition_jvp interpolation_mode dct_coe_num bspline_degree bspline_coe_num <<< "${RUNS[$run_idx]}"
    gpu_group="${GPUS_ARRAY[$gpu_group_idx]}"
    log_file="$(run_path_for train_logs "$seed" "$lambda_flow_k" "$phy_loss_weight" "$image_only_condition_jvp" "$interpolation_mode" "$dct_coe_num" "$bspline_degree" "$bspline_coe_num").log"
    label="seed ${seed}, lambda_flow_k ${lambda_flow_k}, phy_loss_weight ${phy_loss_weight}, image_only_condition_jvp ${image_only_condition_jvp}, interpolation_mode ${interpolation_mode}, dct_coe_num ${dct_coe_num}, bspline_degree ${bspline_degree}, bspline_coe_num ${bspline_coe_num}, gpus ${gpu_group}"

    mkdir -p "$(dirname "$log_file")"
    echo "launch ${label}; log=${log_file}"
    (run_one "$seed" "$gpu_group" "$lambda_flow_k" "$phy_loss_weight" "$image_only_condition_jvp" "$interpolation_mode" "$dct_coe_num" "$bspline_degree" "$bspline_coe_num") >"$log_file" 2>&1 &
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
