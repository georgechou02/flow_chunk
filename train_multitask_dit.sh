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
  # POLICY_USE_JVP_AK='[false]' \
  # POLICY_USE_1_K=false \
  # POLICY_DCT_COE_NUM='[48]' \
  # ENV_EVAL_FREQ=0 \
  # bash train_multitask_dit.sh

  # # 两个独立单卡实验
  # GPUS='0,1' RUN_SEEDS='[1000,1001]' bash train_multitask_dit.sh

  # GPUS='1' \
  # RUN_SEEDS='[1000]' \
  # BATCH_SIZE=64 \
  # STEPS=20000 \
  # NUM_WORKERS=16 \
  # POLICY_LAMBDA_FLOW_K='[0.01]' \
  # POLICY_USE_JVP_AK='[false]' \
  # POLICY_USE_1_K=false \
  # POLICY_DCT_COE_NUM='[40]' \
  # ENV_EVAL_FREQ=20000 \
  # bash train_multitask_dit.sh


  # - 单卡实验，NUM_WORKERS=24 → 总共 24 个
  # - 四卡实验，NUM_WORKERS=24 → 总共 96 个
  # - 两个四卡实验，NUM_WORKERS=24 → 总共 192 个

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

# GPUS="${GPUS:-${GPU_IDS:-${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0 1 2 3 4 5 6}}}}"
GPUS="${GPUS:-${GPU_IDS:-${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0 3 4}}}}"

STEPS="${STEPS:-20000}"
PRE_TRAIN_STEPS="${PRE_TRAIN_STEPS:-${POLICY_PRE_TRAIN_STEPS:-0}}"

# POLICY_LAMBDA_FLOW_K="${POLICY_LAMBDA_FLOW_K:-0.0}"
# POLICY_USE_JVP_AK="${POLICY_USE_JVP_AK:-false}"
# POLICY_USE_1_K="${POLICY_USE_1_K:-false}"
# POLICY_ENABLE_STOCHASTIC="${POLICY_ENABLE_STOCHASTIC:-false}"
# POLICY_DCT_COE_NUM="${POLICY_DCT_COE_NUM:-0}"
POLICY_LAMBDA_FLOW_K="${POLICY_LAMBDA_FLOW_K:-0.01}"
POLICY_USE_JVP_AK="${POLICY_USE_JVP_AK:-false}"
POLICY_USE_1_K="${POLICY_USE_1_K:-true}"
POLICY_ENABLE_STOCHASTIC="${POLICY_ENABLE_STOCHASTIC:-false}"
POLICY_DCT_COE_NUM="${POLICY_DCT_COE_NUM:-48}"

RUN_SEEDS="${RUN_SEEDS:-${SEEDS:-${SEED:-1000 1001 1002}}}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-24}"
# check point
SAVE_FREQ="${SAVE_FREQ:-10000}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-false}"

ENV_EVAL_FREQ="${ENV_EVAL_FREQ:-20000}"
EVAL_SUITES="${EVAL_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-50}"
EVAL_USE_ASYNC_ENVS="${EVAL_USE_ASYNC_ENVS:-true}"
EVAL_OBSERVATION_HEIGHT="${EVAL_OBSERVATION_HEIGHT:-256}"
EVAL_OBSERVATION_WIDTH="${EVAL_OBSERVATION_WIDTH:-256}"

RUN_PREFIX="${RUN_PREFIX:-multitask-dit-flow-libero-lambda-ak-step20000}"
# RUN_PREFIX="${RUN_PREFIX:-multitask-dit-flow-libero-baseline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"

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

if kind == "dct_coe_num":
    normalized_values = []
    for value in values:
        text = str(value).strip()
        if not re.fullmatch(r"[0-9]+", text):
            raise SystemExit(f"dct_coe_num must be a non-negative integer, got {value!r}")
        value = int(text)
        if value > 48:
            raise SystemExit(f"dct_coe_num must not exceed the configured horizon 48, got {value}")
        normalized_values.append(value)
    values = normalized_values

for value in values:
    if kind == "seed":
        value = int(value)
        if value < 0:
            raise SystemExit(f"{kind} must be non-negative")
        print(value)
    elif kind == "lambda_flow_k":
        text = str(value).strip()
        if float(text) < 0:
            raise SystemExit("lambda_flow_k must be non-negative")
        print(text)
    elif kind == "dct_coe_num":
        print(value)
    elif kind == "use_jvp_ak":
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            print("true")
        elif text in {"0", "false", "no", "n", "off"}:
            print("false")
        else:
            raise SystemExit(f"use_jvp_ak must be true or false, got {value!r}")
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
  local use_jvp_ak="$2"
  local dct_coe_num="$3"
  echo "${RUN_PREFIX}_kinematic_$(safe_tag "$lambda_flow_k")_use_ak_$(safe_tag "$use_jvp_ak")_use_1_k_$(safe_tag "$POLICY_USE_1_K")_stochastic_$(safe_tag "$POLICY_ENABLE_STOCHASTIC")_dct_$(safe_tag "$dct_coe_num")_batch_size${BATCH_SIZE}_steps${STEPS}_pre_train_steps${PRE_TRAIN_STEPS}"
}

run_path_for() {
  local kind="$1"
  local seed="$2"
  local lambda_flow_k="$3"
  local use_jvp_ak="$4"
  local dct_coe_num="$5"
  echo "${OUTPUT_ROOT%/}/${kind}/$(run_group_for "$lambda_flow_k" "$use_jvp_ak" "$dct_coe_num")_${RUN_TIMESTAMP}/seed${seed}"
}

run_one() {
  local seed="$1"
  local gpu_group="$2"
  local lambda_flow_k="$3"
  local use_jvp_ak="$4"
  local dct_coe_num="$5"
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

  out="$(run_path_for train "$seed" "$lambda_flow_k" "$use_jvp_ak" "$dct_coe_num")"
  eval_out="$(run_path_for eval "$seed" "$lambda_flow_k" "$use_jvp_ak" "$dct_coe_num")"
  run_group="$(run_group_for "$lambda_flow_k" "$use_jvp_ak" "$dct_coe_num")"
  job_name="${run_group}_seed${seed}"
  wandb_group="${WANDB_GROUP:-$run_group}"
  wandb_notes="${WANDB_NOTES:-LOG_DIR:${wandb_group}}"
  wandb_tags="${WANDB_TAGS:-[\"LOG_DIR:$(safe_tag "$wandb_group")\"]}"

  echo "seed=${seed} gpu=${gpu_group} num_processes=${num_processes} eval_suites=${EVAL_SUITES} lambda_flow_k=${lambda_flow_k} use_jvp_ak=${use_jvp_ak} use_1_k=${POLICY_USE_1_K} enable_stochastic=${POLICY_ENABLE_STOCHASTIC} dct_coe_num=${dct_coe_num} pre_train_steps=${PRE_TRAIN_STEPS} out=${out} eval_out=${eval_out} wandb_group=${wandb_group}"

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
    --policy.pre_train_steps="${PRE_TRAIN_STEPS}"
    --policy.use_jvp_ak="${use_jvp_ak}"
    --policy.use_1_k="${POLICY_USE_1_K}"
    --policy.gripper_first=false
    --policy.enable_stochastic="${POLICY_ENABLE_STOCHASTIC}"
    --policy.sample_frequency=10.0
    --policy.dct_coe_num="${dct_coe_num}"
    --policy.num_integration_steps=32
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
    --steps="${STEPS}"
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq=100
    --output_dir="${out}"
    --policy.push_to_hub=false
  )

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
require_non_negative_int PRE_TRAIN_STEPS
require_non_negative_int SAVE_FREQ
require_non_negative_int ENV_EVAL_FREQ
require_non_negative_int EVAL_EPISODES
require_non_negative_int EVAL_BATCH_SIZE
require_non_negative_int EVAL_OBSERVATION_HEIGHT
require_non_negative_int EVAL_OBSERVATION_WIDTH

mapfile -t SEEDS_ARRAY < <(parse_list "$RUN_SEEDS" seed)
mapfile -t GPUS_ARRAY < <(parse_list "$GPUS" gpu)
mapfile -t LAMBDAS < <(parse_list "$POLICY_LAMBDA_FLOW_K" lambda_flow_k)
mapfile -t USE_JVP_AKS < <(parse_list "$POLICY_USE_JVP_AK" use_jvp_ak)
mapfile -t DCT_COE_NUMS < <(parse_list "$POLICY_DCT_COE_NUM" dct_coe_num)

if (( ${#SEEDS_ARRAY[@]} == 0 || ${#GPUS_ARRAY[@]} == 0 || ${#LAMBDAS[@]} == 0 || ${#USE_JVP_AKS[@]} == 0 || ${#DCT_COE_NUMS[@]} == 0 )); then
  echo "Empty seed/GPU/lambda/use_jvp_ak/dct_coe_num list." >&2
  exit 1
fi

RUNS=()
for seed in "${SEEDS_ARRAY[@]}"; do
  for lambda_flow_k in "${LAMBDAS[@]}"; do
    for use_jvp_ak in "${USE_JVP_AKS[@]}"; do
      for dct_coe_num in "${DCT_COE_NUMS[@]}"; do
        RUNS+=("${seed}|${lambda_flow_k}|${use_jvp_ak}|${dct_coe_num}")
      done
    done
  done
done

NUM_RUNS="${#RUNS[@]}"
NUM_GPU_GROUPS="${#GPUS_ARRAY[@]}"

echo "seeds=${SEEDS_ARRAY[*]}"
echo "gpus=${GPUS_ARRAY[*]}"
echo "batch_size=${BATCH_SIZE}"
echo "steps=${STEPS}"
echo "pre_train_steps=${PRE_TRAIN_STEPS}"
echo "env_eval_freq=${ENV_EVAL_FREQ}"
echo "eval_suites=${EVAL_SUITES}"
echo "eval_episodes=${EVAL_EPISODES}"
echo "eval_batch_size=${EVAL_BATCH_SIZE}"
echo "lambda_flow_k=${LAMBDAS[*]}"
echo "use_jvp_ak=${USE_JVP_AKS[*]}"
echo "use_1_k=${POLICY_USE_1_K}"
echo "enable_stochastic=${POLICY_ENABLE_STOCHASTIC}"
echo "dct_coe_num=${DCT_COE_NUMS[*]}"
echo "output_root=${OUTPUT_ROOT}"
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

    IFS='|' read -r seed lambda_flow_k use_jvp_ak dct_coe_num <<< "${RUNS[$run_idx]}"
    gpu_group="${GPUS_ARRAY[$gpu_group_idx]}"
    log_file="$(run_path_for train_logs "$seed" "$lambda_flow_k" "$use_jvp_ak" "$dct_coe_num").log"
    label="seed ${seed}, lambda_flow_k ${lambda_flow_k}, use_jvp_ak ${use_jvp_ak}, dct_coe_num ${dct_coe_num}, gpus ${gpu_group}"

    mkdir -p "$(dirname "$log_file")"
    echo "launch ${label}; log=${log_file}"
    (run_one "$seed" "$gpu_group" "$lambda_flow_k" "$use_jvp_ak" "$dct_coe_num") >"$log_file" 2>&1 &
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
