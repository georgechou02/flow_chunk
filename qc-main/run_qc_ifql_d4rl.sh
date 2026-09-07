#!/usr/bin/env bash
# D4RL 小样本 QC（best-of-n flow actor + 速度场 JVP 高阶约束）。
# 论文 QC，不是 QC-FQL，也不是 QC-IFQL。文件名 run_qc_ifql_d4rl.sh 是历史命名。
# 环境和 subsample ratio 对齐 fql-master/run_ifql_d4rl.sh。
# 用法（在 d4rl conda 环境里）：
#   conda activate d4rl
#   cd qc-main
#   nohup bash run_qc_ifql_d4rl.sh > logs/run_qc_ifql_d4rl.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

export XLA_PYTHON_CLIENT_PREALLOCATE=False

mkdir -p "${SCRIPT_DIR}/logs"

ENV_NAMES=(
    # "hopper-medium-v2"
    "hopper-medium-replay-v2"
    "hopper-medium-expert-v2"
    # "hopper-expert-v2"
    # "halfcheetah-medium-v2"
    "halfcheetah-medium-replay-v2"
    "halfcheetah-medium-expert-v2"
    # "halfcheetah-expert-v2"
    # "walker2d-medium-v2"
    "walker2d-medium-replay-v2"
    "walker2d-medium-expert-v2"
    # "walker2d-expert-v2"
    # "antmaze-umaze-v2"
    # "antmaze-umaze-diverse-v2"
    # "antmaze-medium-play-v2"
    # "antmaze-medium-diverse-v2"
    # "antmaze-large-play-v2"
    # "antmaze-large-diverse-v2"
    # "pen-human-v1"
    # "pen-cloned-v1"
    # "pen-expert-v1"
    # "door-human-v1"
    # "door-cloned-v1"
    # "door-expert-v1"
    # "hammer-human-v1"
    # "hammer-cloned-v1"
    # "hammer-expert-v1"
    # "relocate-human-v1"
    # "relocate-cloned-v1"
    # "relocate-expert-v1"
)

# lambda 扫描对齐 fql-master 的 kinematic 实验；QC 没有 critic grad_weights。
LAMBDA_FLOW_KS=(0.1 0.5 1)
# 默认只跑前向差分，对齐 flow actor 的 a' - a。需要 DCT 时加上 dct。
DERIV_MODES=(forward dct)
SEEDS=(0 1 2)

HORIZON_LENGTH=5
OFFLINE_STEPS=1000000
ONLINE_STEPS=0
EVAL_INTERVAL=100000
EVAL_EPISODES=50
# wandb 前缀是历史命名（内容是 QC / best-of-n，不是论文 QC-IFQL）。
RUN_GROUP_PREFIX="qc_ifql_d4rl"
PROJECT="QC_IFQL-Locomotion"

# OGBench kinematic 占用 2/3/4，这里用空闲卡。
GPUS=(5 6 7)
JOBS_PER_GPU=4

declare -A GPU_PIDS
for gpu in "${GPUS[@]}"; do
  GPU_PIDS[$gpu]=""
done

update_active_jobs() {
  local gpu=$1
  local active=0
  local new_pids=""

  for pid in ${GPU_PIDS[$gpu]}; do
    if ps -p "$pid" > /dev/null 2>&1; then
      active=$((active + 1))
      new_pids="$new_pids $pid"
    fi
  done

  GPU_PIDS[$gpu]="$new_pids"
  ACTIVE_JOBS=$active
}

wait_for_slot() {
  local assigned_gpu=""
  while true; do
    assigned_gpu=""
    for gpu_id in "${GPUS[@]}"; do
      update_active_jobs "$gpu_id"
      if [ "$ACTIVE_JOBS" -lt "${JOBS_PER_GPU}" ]; then
        assigned_gpu=$gpu_id
        break
      fi
    done
    if [ -n "$assigned_gpu" ]; then
      echo "$assigned_gpu"
      return 0
    fi
    sleep 2
  done
}

d4rl_ratio() {
  local env_name=$1
  case "$env_name" in
    antmaze-umaze-v2|antmaze-umaze-diverse-v2)
      echo 100
      ;;
    antmaze-medium-*|antmaze-large-*)
      echo 10
      ;;
    *-medium-expert-v2)
      echo 200
      ;;
    *-human-v1)
      echo 1
      ;;
    halfcheetah-medium-replay-v2)
      echo 20
      ;;
    hopper-medium-replay-v2)
      echo 40
      ;;
    walker2d-medium-replay-v2)
      echo 30
      ;;
    pen-cloned-v1|pen-expert-v1|door-cloned-v1|door-expert-v1|hammer-cloned-v1|hammer-expert-v1|relocate-cloned-v1|relocate-expert-v1)
      echo 50
      ;;
    *)
      echo 100
      ;;
  esac
}

env_extra_args() {
  local env_name=$1
  case "$env_name" in
    *antmaze*)
      echo "--discount=0.995 --agent.discount=0.995"
      ;;
    *cloned*|*human*)
      echo "--agent.actor_num_samples=128"
      ;;
    *)
      echo "--agent.actor_num_samples=32"
      ;;
  esac
}

counter=0
total=$((${#ENV_NAMES[@]} * ${#LAMBDA_FLOW_KS[@]} * ${#DERIV_MODES[@]} * ${#SEEDS[@]}))
echo "Planning ${total} QC D4RL jobs on GPUs ${GPUS[*]} (${JOBS_PER_GPU} per GPU)."

for ENV_NAME in "${ENV_NAMES[@]}"; do
  RATIO="$(d4rl_ratio "${ENV_NAME}")"
  EXTRA_ARGS="$(env_extra_args "${ENV_NAME}")"
  for lambda_flow_k in "${LAMBDA_FLOW_KS[@]}"; do
    for deriv_mode in "${DERIV_MODES[@]}"; do
      if [ "${deriv_mode}" = "forward" ]; then
        DCT_COE_NUM=0
      else
        DCT_COE_NUM="${HORIZON_LENGTH}"
      fi
      for seed in "${SEEDS[@]}"; do
        gpu_id="$(wait_for_slot)"
        run_group="${RUN_GROUP_PREFIX}_${deriv_mode}_l${lambda_flow_k}"
        log_file="${SCRIPT_DIR}/logs/qc_ifql_${ENV_NAME}_${deriv_mode}_l${lambda_flow_k}_seed${seed}.log"

        echo "[$((counter + 1))/${total}] GPU ${gpu_id}: ${ENV_NAME} ratio=${RATIO} mode=${deriv_mode} lambda_flow_k=${lambda_flow_k} seed=${seed} ${EXTRA_ARGS}"

        (
          cd "${SCRIPT_DIR}"
          CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" main.py \
            --project="${PROJECT}" \
            --run_group="${run_group}" \
            --env_name="${ENV_NAME}" \
            --ratio="${RATIO}" \
            --horizon_length="${HORIZON_LENGTH}" \
            --offline_steps="${OFFLINE_STEPS}" \
            --online_steps="${ONLINE_STEPS}" \
            --eval_interval="${EVAL_INTERVAL}" \
            --eval_episodes="${EVAL_EPISODES}" \
            --seed="${seed}" \
            --agent.actor_type=best-of-n \
            --agent.action_chunking=True \
            --agent.q_agg=min \
            --agent.lambda_flow_k="${lambda_flow_k}" \
            --agent.dct_coe_num="${DCT_COE_NUM}" \
            ${EXTRA_ARGS}
        ) > "${log_file}" 2>&1 &

        new_pid=$!
        GPU_PIDS[$gpu_id]="${GPU_PIDS[$gpu_id]} $new_pid"
        echo "  started PID ${new_pid}  log=${log_file}"

        counter=$((counter + 1))
        sleep 0.5
      done
    done
  done
done

wait
echo "All ${counter} QC D4RL jobs completed."

# nohup bash run_qc_ifql_d4rl.sh > logs/run_qc_ifql_d4rl.log 2>&1 &