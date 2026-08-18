#!/usr/bin/env bash
# OGBench 纯 offline QC-IFQL（best-of-n）+ 高阶约束（速度场 JVP，对齐 DiT / IFQL actor）：
#   cube-double task1/2、scene-sparse task2/3
#   lambda_flow_k in {0.01, 0.1} × {forward, DCT} × 3 seeds
# 用法（在 d4rl conda 环境里）：
#   conda activate d4rl
#   cd qc-main
#   nohup bash run_ogbench_kinematic.sh > logs/run_ogbench_kinematic.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

export XLA_PYTHON_CLIENT_PREALLOCATE=False

mkdir -p "${SCRIPT_DIR}/logs"

# name|env_name|sparse
TASKS=(
  "cube-double-task1|cube-double-play-singletask-task1-v0|False"
  "cube-double-task2|cube-double-play-singletask-task2-v0|False"
  "scene-sparse-task2|scene-play-singletask-task2-v0|True"
  "scene-sparse-task3|scene-play-singletask-task3-v0|True"
)

LAMBDA_FLOW_KS=(0.01 0.1)
# dct_coe_num=0: 前向差分; dct_coe_num=horizon: 截断 DCT
DERIV_MODES=(forward dct)
SEEDS=(0 1 2)

HORIZON_LENGTH=5
OFFLINE_STEPS=1000000
ONLINE_STEPS=0
EVAL_INTERVAL=100000
EVAL_EPISODES=50
RUN_GROUP_PREFIX="kin_ifql"

GPUS=(2 3 4)
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

counter=0
total=$((${#TASKS[@]} * ${#LAMBDA_FLOW_KS[@]} * ${#DERIV_MODES[@]} * ${#SEEDS[@]}))
echo "Planning ${total} jobs on GPUs ${GPUS[*]} (${JOBS_PER_GPU} per GPU)."

for task in "${TASKS[@]}"; do
  IFS='|' read -r TASK_NAME ENV_NAME SPARSE <<< "${task}"
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
        log_file="${SCRIPT_DIR}/logs/${TASK_NAME}_${deriv_mode}_l${lambda_flow_k}_seed${seed}.log"

        echo "[$((counter + 1))/${total}] GPU ${gpu_id}: ${TASK_NAME} mode=${deriv_mode} dct_coe_num=${DCT_COE_NUM} lambda_flow_k=${lambda_flow_k} seed=${seed}"

        (
          cd "${SCRIPT_DIR}"
          CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" main.py \
            --run_group="${run_group}" \
            --env_name="${ENV_NAME}" \
            --sparse="${SPARSE}" \
            --horizon_length="${HORIZON_LENGTH}" \
            --offline_steps="${OFFLINE_STEPS}" \
            --online_steps="${ONLINE_STEPS}" \
            --eval_interval="${EVAL_INTERVAL}" \
            --eval_episodes="${EVAL_EPISODES}" \
            --seed="${seed}" \
            --agent.actor_type=best-of-n \
            --agent.actor_num_samples=32 \
            --agent.action_chunking=True \
            --agent.lambda_flow_k="${lambda_flow_k}" \
            --agent.dct_coe_num="${DCT_COE_NUM}"
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
echo "All ${counter} QC-IFQL kinematic offline jobs completed."
