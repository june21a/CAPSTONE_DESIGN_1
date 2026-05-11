#!/bin/bash

# env variables
# export CARLA_ROOT="/home/carla"
# export WORK_DIR="/home/carla_garage"
# export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
# export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
# export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
# export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# ===== Path Settings =====
CARLA_GARAGE_DIR="/home/carla_garage"
AGENT_CONFIG="./pretrained_models/all_towns"
AGENT="./team_code/sensor_agent.py"
ROUTES="./leaderboard/data/bench2drive220.xml"
SAVE_PATH_DIR="./results/bench2drive220"
CHECKPOINT="${SAVE_PATH_DIR}/debug_results.json"

# ===== Environment Variables =====
export DEBUG_CHALLENGE=1
export SAVE_PATH="${SAVE_PATH_DIR}"

# ===== Move Directory =====
cd "${CARLA_GARAGE_DIR}" || exit

# ===== Run =====
python ./leaderboard/leaderboard/leaderboard_evaluator_local.py \
    --agent-config "${AGENT_CONFIG}" \
    --agent "${AGENT}" \
    --routes "${ROUTES}" \
    --checkpoint "${CHECKPOINT}"