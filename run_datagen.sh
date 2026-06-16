#!/bin/bash

# ===== CARLA Environment =====
CURRENT_DIR=$(pwd)
export CARLA_ROOT="${CURRENT_DIR}/carla_garage/carla"
export WORK_DIR="${CURRENT_DIR}/carla_garage"

export SCENARIO_RUNNER_ROOT="${WORK_DIR}/scenario_runner_autopilot"
export LEADERBOARD_ROOT="${WORK_DIR}/leaderboard_autopilot"

# ===== PYTHONPATH =====
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

# ===== Path Settings =====
CARLA_GARAGE_DIR="${CURRENT_DIR}/carla_garage"
AGENT_CONFIG="/home/ec2-user/AD_challenge/experiments/pretrained_baseline"
AGENT="./team_code/transfuser_datagen_agent.py"
ROUTES="/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive220_pretrained_baseline_training_data.xml"
SAVE_PATH_DIR="./results/pretrained_baseline_training_data_freq1"
CHECKPOINT="${SAVE_PATH_DIR}/debug_results.json"
RESUME=1

# ===== Environment Variables =====
export DEBUG_CHALLENGE=1
export DATAGEN=1
export SAVE_PATH="${SAVE_PATH_DIR}"
export DATAGEN_RENAME_ROUTE_FOLDER=1
export COLLECT_SENSOR_DATA=0
export ATTENTION_VIS=0
export VISION_TASK_VIS=0
export ATTENTION_SAVE_FREQ=1
export DATA_SAVE_FREQ=1
export DELETE_ROUTE_FOLDER_WITHOUT_COLLISION=0
export GENERATE_PLAN_SAFETY_LABELS=0
export DISABLE_CUDNN=0
export FORCE_CPU=0

# ===== Move Directory =====
cd "${CARLA_GARAGE_DIR}" || exit

# ===== Run =====
exec python ./leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py \
    --agent-config "${AGENT_CONFIG}" \
    --agent "${AGENT}" \
    --routes "${ROUTES}" \
    --checkpoint "${CHECKPOINT}" \
    --resume "${RESUME}"
