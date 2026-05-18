#!/bin/bash

# ===== CARLA Environment =====
CURRENT_DIR=$(pwd)
export CARLA_ROOT="${CURRENT_DIR}/carla_garage/carla"
export WORK_DIR="${CURRENT_DIR}/carla_garage"

export SCENARIO_RUNNER_ROOT="${WORK_DIR}/scenario_runner"
export LEADERBOARD_ROOT="${WORK_DIR}/leaderboard"

# ===== PYTHONPATH =====
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

# ===== Path Settings =====
CARLA_GARAGE_DIR="${CURRENT_DIR}/carla_garage"
AGENT_CONFIG="./pretrained_models/all_towns"
AGENT="./team_code/comparison_agent.py"
ROUTES="./leaderboard/data/bench2drive220.xml"
SAVE_PATH_DIR="./results/bench2drive220"
CHECKPOINT="${SAVE_PATH_DIR}/debug_results.json"
RESUME=1

# ===== Environment Variables =====
export DEBUG_CHALLENGE=1
export SAVE_PATH="${SAVE_PATH_DIR}"
export COLLECT_SENSOR_DATA=1
export ATTENTION_VIS=1
export VISION_TASK_VIS=1
export ATTENTION_SAVE_FREQ=1
export DISABLE_CUDNN=0
export FORCE_CPU=0

# ===== Move Directory =====
cd "${CARLA_GARAGE_DIR}" || exit

# ===== Run =====
python ./leaderboard/leaderboard/leaderboard_evaluator_local.py \
    --agent-config "${AGENT_CONFIG}" \
    --agent "${AGENT}" \
    --routes "${ROUTES}" \
    --checkpoint "${CHECKPOINT}" \
    --resume "${RESUME}"