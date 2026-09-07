#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhouzhi/code_store/higher-order/lerobot}"
DATASET_ROOT="${DATASET_ROOT:-/home/zhouzhi/code_store/higher-order/PutSausageInPot-joint}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/PutSausageInPot-joint}"
NUM_TRAJ="${NUM_TRAJ:-30}"

DATASET_EPISODES=""
if [[ "$NUM_TRAJ" == "-1" ]]; then
  NUM_TRAJ_LABEL="all"
elif [[ "$NUM_TRAJ" =~ ^[1-9][0-9]*$ ]]; then
  NUM_TRAJ_LABEL="$NUM_TRAJ"
  episode_indices=()
  for ((episode_idx = 0; episode_idx < NUM_TRAJ; episode_idx++)); do
    episode_indices+=("$episode_idx")
  done
  printf -v joined_episode_indices ',%s' "${episode_indices[@]}"
  DATASET_EPISODES="[${joined_episode_indices:1}]"
else
  echo "NUM_TRAJ must be -1 (all episodes) or a positive integer, got: $NUM_TRAJ" >&2
  exit 1
fi

ACCELERATE_BIN="${ACCELERATE_BIN:-/data/zhouzhi/conda_envs/lerobot/bin/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-/data/zhouzhi/conda_envs/lerobot/bin/lerobot-train}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ARRAY[@]}}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEVICE="${DEVICE:-cuda}"

SEED="${SEED:-1000}"
STEPS="${STEPS:-10000}"
# single
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SAVE_FREQ="${SAVE_FREQ:-5000}"

PRE_TRAIN_STEPS="${PRE_TRAIN_STEPS:-0}"
LAMBDA_FLOW_K="${LAMBDA_FLOW_K:-0.01}"
DCT_COE_NUM="${DCT_COE_NUM:-10}"
DERIVATIVE_MODE="${DERIVATIVE_MODE:-central}"
IMAGE_ONLY_CONDITION_JVP="${IMAGE_ONLY_CONDITION_JVP:-true}"

WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-multitask_dit_real}"
WANDB_MODE="${WANDB_MODE:-offline}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_NAME="${RUN_NAME:-PutSausageInPot-num_traj${NUM_TRAJ_LABEL}-clip-${DERIVATIVE_MODE}-image_only_${IMAGE_ONLY_CONDITION_JVP}-lambda${LAMBDA_FLOW_K}-seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/real_robot}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}_${RUN_TIMESTAMP}}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TMPDIR="${TMPDIR:-/data/zhouzhi/tmp}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${TMPDIR}/hf_datasets_cache_flow}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR}/numba_cache}"

if [[ ! -x "$ACCELERATE_BIN" ]]; then
  echo "Missing accelerate executable: $ACCELERATE_BIN" >&2
  exit 1
fi
if [[ ! -x "$LEROBOT_TRAIN_BIN" ]]; then
  echo "Missing lerobot-train executable: $LEROBOT_TRAIN_BIN" >&2
  exit 1
fi
if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "Invalid LeRobot dataset root: $DATASET_ROOT" >&2
  exit 1
fi
total_episodes_pattern='"total_episodes"[[:space:]]*:[[:space:]]*([0-9]+)'
TOTAL_EPISODES=""
while IFS= read -r metadata_line; do
  if [[ "$metadata_line" =~ $total_episodes_pattern ]]; then
    TOTAL_EPISODES="${BASH_REMATCH[1]}"
    break
  fi
done < "$DATASET_ROOT/meta/info.json"
if [[ -z "$TOTAL_EPISODES" ]]; then
  echo "Could not read total_episodes from: $DATASET_ROOT/meta/info.json" >&2
  exit 1
fi
if [[ "$NUM_TRAJ" != "-1" ]] && ((NUM_TRAJ > TOTAL_EPISODES)); then
  echo "NUM_TRAJ=$NUM_TRAJ exceeds dataset total_episodes=$TOTAL_EPISODES" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "Output path already exists: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$NUMBA_CACHE_DIR"
cd "$REPO_ROOT"

INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[14]},"observation.images.global_front":{"type":"VISUAL","shape":[3,480,848]},"observation.images.wrist_left":{"type":"VISUAL","shape":[3,480,640]},"observation.images.wrist_right":{"type":"VISUAL","shape":[3,480,640]}}'
IMAGE_TRANSFORMS='{"brightness":{"type":"ColorJitter","kwargs":{"brightness":[0.75,1.25]}},"contrast":{"type":"ColorJitter","kwargs":{"contrast":[0.6,1.4]}},"saturation":{"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}},"hue":{"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"type":"SharpnessJitter","kwargs":{"sharpness":[0.6,1.4]}},"rotation":{"type":"RandomRotation","kwargs":{"degrees":[-5,5]}},"translation":{"type":"RandomAffine","kwargs":{"degrees":0,"translate":[0.1,0.1]}}}'

cmd=(
  "$ACCELERATE_BIN" launch
  --num_processes="$NUM_PROCESSES"
  --mixed_precision="$MIXED_PRECISION"
)
if (( NUM_PROCESSES > 1 )); then
  cmd+=(--multi_gpu --main_process_port=0)
fi

cmd+=(
  "$LEROBOT_TRAIN_BIN"
  --job_name="$RUN_NAME"
  --resume=false
  --seed="$SEED"
  --wandb.enable="$WANDB_ENABLE"
  --wandb.project="$WANDB_PROJECT"
  --wandb.mode="$WANDB_MODE"
  --policy.type=multi_task_dit
  --policy.device="$DEVICE"
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
  --policy.use_separate_rgb_encoder_per_camera=false
  '--policy.image_resize_shape=[256,256]'
  '--policy.image_crop_shape=[224,224]'
  --policy.image_crop_is_random=true
  --policy.text_encoder_name=openai/clip-vit-base-patch16
  --policy.do_mask_loss_for_padding=false
  --policy.sigma_min=0.0
  --policy.lambda_flow_k="$LAMBDA_FLOW_K"
  --policy.pre_train_steps="$PRE_TRAIN_STEPS"
  --policy.gripper_first=false
  --policy.sample_frequency=30.0
  --policy.dct_coe_num="$DCT_COE_NUM"
  --policy.conditioning_derivative_mode="$DERIVATIVE_MODE"
  --policy.image_only_condition_jvp="$IMAGE_ONLY_CONDITION_JVP"
  --policy.num_integration_steps=32
  --policy.integration_method=euler
  --policy.timestep_sampling_strategy=beta
  --policy.timestep_sampling_alpha=1.5
  --policy.timestep_sampling_beta=1.0
  --policy.timestep_sampling_s=0.999
  --policy.input_features="$INPUT_FEATURES"
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.use_imagenet_stats=true
  --dataset.image_transforms.enable=true
  --dataset.image_transforms.max_num_transforms=4
  --dataset.image_transforms.tfs="$IMAGE_TRANSFORMS"
  --dataset.video_backend=torchcodec
  --env_eval_freq=0
  --steps="$STEPS"
  --batch_size="$BATCH_SIZE"
  --num_workers="$NUM_WORKERS"
  --save_freq="$SAVE_FREQ"
  --log_freq=100
  --output_dir="$OUTPUT_DIR"
  --policy.push_to_hub=false
)

if [[ -n "$DATASET_EPISODES" ]]; then
  cmd+=(--dataset.episodes="$DATASET_EPISODES")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" "${cmd[@]}"



# NUM_TRAJ=30 LAMBDA_FLOW_K=0.01 bash pipeline/scripts/run_training.sh && \
# NUM_TRAJ=30 LAMBDA_FLOW_K=0 bash pipeline/scripts/run_training.sh
