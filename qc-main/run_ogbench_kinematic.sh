#!/usr/bin/env bash
# OGBench 纯 offline QC（best-of-n flow actor）+ 高阶约束（速度场 JVP，对齐 DiT）：
#   论文 QC，不是 QC-FQL（distill-ddpg），也不是 QC-IFQL（chunked IQL + rejection）。
#   cube-double / cube-triple
#   lambda_flow_k × derivative estimator × 3 seeds; physical weight = 0
# wandb group: kin_ifql_{kind}_l{lambda}_p{phy}_ak{jvp}（历史前缀，内容是 QC）
# 满模态：DCT / Chebyshev / B-spline 的 M=H；Savitzky–Golay 无模态数，用 window=3,p=1（内部中心差分）。
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
  # "cube-double-task3|cube-double-play-singletask-task3-v0|False"
  # "cube-double-task4|cube-double-play-singletask-task4-v0|False"
  # "cube-triple-task1|cube-triple-play-singletask-task1-v0|False"
  "cube-triple-task2|cube-triple-play-singletask-task2-v0|False"
  "cube-triple-task3|cube-triple-play-singletask-task3-v0|False"
  # "cube-triple-task4|cube-triple-play-singletask-task4-v0|False"
)

LAMBDA_FLOW_KS=(0 0.01 0.001)
PHY_LOSS_WEIGHTS=(0)  # Compare QC vs QC + FOS only; no physical loss.
ACTION_JVP_GRAD_SCALES=(1.0)
# 满模态：DCT / Chebyshev / B-spline 的 M=H；Savitzky–Golay 无模态，window=3 等价于最少平滑。
# DERIV_MODES=(forward dct savgol bspline chebyshev)
DERIV_MODES=(savgol bspline dct)
SEEDS=(0 1 2)

HORIZON_LENGTH=5
OFFLINE_STEPS=1000000
ONLINE_STEPS=0
EVAL_INTERVAL=100000
EVAL_EPISODES=10
# wandb 前缀是历史命名（内容是 QC / best-of-n，不是论文 QC-IFQL）。
RUN_GROUP_PREFIX="kin_ifql"

GPUS=(4 5 6 7)
JOBS_PER_GPU=8

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

is_zero() {
  [ "$1" = "0" ] || [ "$1" = "0.0" ]
}

should_skip_combo() {
  local lambda_flow_k="$1"
  local phy_loss_weight="$2"
  local deriv_mode="$4"
  # A single vanilla QC run per task/seed; its unused derivative estimator
  # must not create duplicate baseline runs.
  if is_zero "${lambda_flow_k}" && is_zero "${phy_loss_weight}"; then
    [ "${deriv_mode}" != "${DERIV_MODES[0]}" ]
    return
  fi
  return 1
}

counter=0
total=0
for task in "${TASKS[@]}"; do
  for lambda_flow_k in "${LAMBDA_FLOW_KS[@]}"; do
    for phy_loss_weight in "${PHY_LOSS_WEIGHTS[@]}"; do
      for action_jvp_grad_scale in "${ACTION_JVP_GRAD_SCALES[@]}"; do
        for deriv_mode in "${DERIV_MODES[@]}"; do
          if should_skip_combo "${lambda_flow_k}" "${phy_loss_weight}" "${action_jvp_grad_scale}" "${deriv_mode}"; then
            continue
          fi
          total=$((total + ${#SEEDS[@]}))
        done
      done
    done
  done
done
echo "Planning ${total} jobs on GPUs ${GPUS[*]} (${JOBS_PER_GPU} per GPU)."
echo "lambda_flow_k=${LAMBDA_FLOW_KS[*]} phy_loss_weight=${PHY_LOSS_WEIGHTS[*]} action_jvp_grad_scale=${ACTION_JVP_GRAD_SCALES[*]}"

for task in "${TASKS[@]}"; do
  IFS='|' read -r TASK_NAME ENV_NAME SPARSE <<< "${task}"
  for lambda_flow_k in "${LAMBDA_FLOW_KS[@]}"; do
    for phy_loss_weight in "${PHY_LOSS_WEIGHTS[@]}"; do
      for action_jvp_grad_scale in "${ACTION_JVP_GRAD_SCALES[@]}"; do
        for deriv_mode in "${DERIV_MODES[@]}"; do
          if should_skip_combo "${lambda_flow_k}" "${phy_loss_weight}" "${action_jvp_grad_scale}" "${deriv_mode}"; then
            continue
          fi
          KIND_ARGS=(--agent.derivative_kind="${deriv_mode}")
          group_kind="${deriv_mode}"
          if is_zero "${lambda_flow_k}" && is_zero "${phy_loss_weight}"; then
            # Vanilla QC: estimator is unused; label the run as baseline.
            group_kind="baseline"
            KIND_ARGS=(--agent.derivative_kind=forward --agent.dct_coe_num=0)
          else
            case "${deriv_mode}" in
              forward)
                KIND_ARGS+=(--agent.dct_coe_num=0)
                ;;
              dct)
                KIND_ARGS+=(--agent.dct_coe_num="${HORIZON_LENGTH}")
                ;;
              savgol)
                KIND_ARGS+=(
                  --agent.savgol_window=5
                  --agent.savgol_polyorder=2
                )
                ;;
              bspline)
                KIND_ARGS+=(
                  --agent.bspline_coe_num="${HORIZON_LENGTH}"
                  --agent.bspline_degree=2
                )
                ;;
              chebyshev)
                KIND_ARGS+=(--agent.chebyshev_num_modes="${HORIZON_LENGTH}")
                ;;
            esac
          fi
          for seed in "${SEEDS[@]}"; do
            gpu_id="$(wait_for_slot)"
            run_group="${RUN_GROUP_PREFIX}_${group_kind}_l${lambda_flow_k}_p${phy_loss_weight}_ak${action_jvp_grad_scale}"
            log_file="${SCRIPT_DIR}/logs/${TASK_NAME}_${group_kind}_l${lambda_flow_k}_p${phy_loss_weight}_ak${action_jvp_grad_scale}_seed${seed}.log"

            echo "[$((counter + 1))/${total}] GPU ${gpu_id}: ${TASK_NAME} kind=${group_kind} lambda_flow_k=${lambda_flow_k} phy_loss_weight=${phy_loss_weight} action_jvp_grad_scale=${action_jvp_grad_scale} seed=${seed} args=${KIND_ARGS[*]}"

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
                --agent.phy_loss_weight="${phy_loss_weight}" \
                --agent.action_jvp_grad_scale="${action_jvp_grad_scale}" \
                "${KIND_ARGS[@]}"
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
  done
done

wait
echo "All ${counter} QC kinematic offline jobs completed."

# nohup bash run_ogbench_kinematic.sh > logs/run_ogbench_kinematic.log 2>&1 &
