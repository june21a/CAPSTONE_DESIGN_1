#!/bin/bash
set -euo pipefail

# ===== CARLA Environment =====
CURRENT_DIR=$(pwd)
export CARLA_ROOT="${CURRENT_DIR}/carla_garage/carla"
export WORK_DIR="${CURRENT_DIR}/carla_garage"
export SCENARIO_RUNNER_ROOT="${WORK_DIR}/scenario_runner_autopilot"
export LEADERBOARD_ROOT="${WORK_DIR}/leaderboard_autopilot"
export PYTHON_BIN="${PYTHON_BIN:-python}"

# ===== PYTHONPATH =====
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CARLA_ROOT}/PythonAPI/carla/dist:${CARLA_ROOT}/LibCarla/lib:${LD_LIBRARY_PATH:-}"

# ===== Defaults =====
export FREE_WORLD_PORT="${FREE_WORLD_PORT:-2000}"
export TM_PORT="${TM_PORT:-8000}"
export SIM_FAILURE_COLLISION_TAIL_SECONDS="${SIM_FAILURE_COLLISION_TAIL_SECONDS:-10}"
export LOG_DIR="${CURRENT_DIR}/logs"

# ===== CARLA Server Lifecycle =====
CARLA_MANAGE_SERVER="${CARLA_MANAGE_SERVER:-1}"
CARLA_RESTART_EVERY="${CARLA_RESTART_EVERY:-6}"
CARLA_STARTUP_WAIT="${CARLA_STARTUP_WAIT:-40}"
CARLA_RESTART_WAIT="${CARLA_RESTART_WAIT:-40}"
DATAGEN_RUN_COUNT=0
CARLA_PID=""
CARLA_STARTED_BY_SCRIPT=0

mkdir -p "${LOG_DIR}"

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $*"
}

check_carla() {
    "${PYTHON_BIN}" - <<'PY'
import os
import sys
import carla

try:
    client = carla.Client("localhost", int(os.environ.get("FREE_WORLD_PORT", "2000")))
    client.set_timeout(5.0)
    client.get_world()
except Exception as exc:
    print(exc)
    sys.exit(1)
PY
}

start_carla() {
    if [[ "${CARLA_MANAGE_SERVER}" != "1" ]]; then
        return
    fi

    if check_carla >/dev/null 2>&1; then
        log "Using existing CARLA server on port ${FREE_WORLD_PORT}; this script will not stop it"
        CARLA_PID=""
        CARLA_STARTED_BY_SCRIPT=0
        return
    fi

    local log_file="${LOG_DIR}/carla_sim_failure_pairs_$(date +%Y%m%d_%H%M%S).log"
    log "Starting CARLA server. Log: ${log_file}"
    setsid bash -lc "
    cd '${CARLA_ROOT}' || exit 1
    export XDG_RUNTIME_DIR=/tmp/runtime-\$(whoami)
    mkdir -p \"\$XDG_RUNTIME_DIR\"
    chmod 700 \"\$XDG_RUNTIME_DIR\"
    exec ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=${FREE_WORLD_PORT}
" >"${log_file}" 2>&1 &

    CARLA_PID=$!
    CARLA_STARTED_BY_SCRIPT=1
    log "CARLA PID: ${CARLA_PID}"
    log "Waiting ${CARLA_STARTUP_WAIT}s before running datagen"
    sleep "${CARLA_STARTUP_WAIT}"
}

stop_carla() {
    if [[ "${CARLA_MANAGE_SERVER}" != "1" ]]; then
        return
    fi

    if [[ "${CARLA_STARTED_BY_SCRIPT}" -eq 1 && -n "${CARLA_PID}" ]] && kill -0 "${CARLA_PID}" 2>/dev/null; then
        log "Stopping CARLA PID ${CARLA_PID}"
        kill -9 "${CARLA_PID}" 2>/dev/null || true
        sleep "${CARLA_RESTART_WAIT}"
    fi
    CARLA_PID=""
    CARLA_STARTED_BY_SCRIPT=0
}

restart_carla() {
    if [[ "${CARLA_MANAGE_SERVER}" != "1" ]]; then
        return
    fi

    log "Restarting CARLA after ${DATAGEN_RUN_COUNT} datagen runs"
    stop_carla
    start_carla
}

after_datagen_run() {
    DATAGEN_RUN_COUNT=$((DATAGEN_RUN_COUNT + 1))
    if [[ "${CARLA_RESTART_EVERY}" -gt 0 && $((DATAGEN_RUN_COUNT % CARLA_RESTART_EVERY)) -eq 0 ]]; then
        restart_carla
    fi
}

cleanup() {
    stop_carla
}

trap cleanup EXIT INT TERM

cd "${WORK_DIR}"
start_carla

# bench2drive_split/bench2drive_103 success
echo "Running bench2drive_split/bench2drive_103 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=42 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_103.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_103_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_103.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_103 failure
echo "Running bench2drive_split/bench2drive_103 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=43 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_103.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_103_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_103.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_42 success
echo "Running bench2drive_split/bench2drive_42 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=44 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_42.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_42_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_42.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_42 failure
echo "Running bench2drive_split/bench2drive_42 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=45 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_42.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_42_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_42.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_49 success
echo "Running bench2drive_split/bench2drive_49 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=46 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_49.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_49_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_49.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_49 failure
echo "Running bench2drive_split/bench2drive_49 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=47 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_49.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_49_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_49.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_46 success
echo "Running bench2drive_split/bench2drive_46 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=48 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_46.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_46_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_46.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_46 failure
echo "Running bench2drive_split/bench2drive_46 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=49 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_46.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_46_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_46.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_205 success
echo "Running bench2drive_split/bench2drive_205 success"
DATAGEN=1 TOWN=Town11 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=50 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_205.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_205_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_205.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_205 failure
echo "Running bench2drive_split/bench2drive_205 failure"
DATAGEN=1 TOWN=Town11 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=51 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_205.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_205_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_205.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_113 success
echo "Running bench2drive_split/bench2drive_113 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=52 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_113.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_113_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_113.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_113 failure
echo "Running bench2drive_split/bench2drive_113 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=53 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_113.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_113_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_113.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_193 success
echo "Running bench2drive_split/bench2drive_193 success"
DATAGEN=1 TOWN=Town07 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=54 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_193.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_193_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_193.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_193 failure
echo "Running bench2drive_split/bench2drive_193 failure"
DATAGEN=1 TOWN=Town07 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=55 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_193.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_193_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_193.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_174 success
echo "Running bench2drive_split/bench2drive_174 success"
DATAGEN=1 TOWN=Town05 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=56 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_174.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_174_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_174.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_174 failure
echo "Running bench2drive_split/bench2drive_174 failure"
DATAGEN=1 TOWN=Town05 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=57 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_174.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_174_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_174.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_51 success
echo "Running bench2drive_split/bench2drive_51 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=58 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_51.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_51_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_51.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_51 failure
echo "Running bench2drive_split/bench2drive_51 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=59 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_51.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_51_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_51.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_214 success
echo "Running bench2drive_split/bench2drive_214 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=60 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_214.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_214_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_214.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_214 failure
echo "Running bench2drive_split/bench2drive_214 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=61 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_214.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_214_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_214.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_202 success
echo "Running bench2drive_split/bench2drive_202 success"
DATAGEN=1 TOWN=Town03 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=62 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_202.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_202_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_202.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_202 failure
echo "Running bench2drive_split/bench2drive_202 failure"
DATAGEN=1 TOWN=Town03 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=63 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_202.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_202_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_202.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_120 success
echo "Running bench2drive_split/bench2drive_120 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=64 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_120.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_120_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_120.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_120 failure
echo "Running bench2drive_split/bench2drive_120 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=65 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_120.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_120_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_120.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_135 success
echo "Running bench2drive_split/bench2drive_135 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=66 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_135.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_135_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_135.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_135 failure
echo "Running bench2drive_split/bench2drive_135 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=67 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_135.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_135_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_135.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_218 success
echo "Running bench2drive_split/bench2drive_218 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=68 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_218.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_218_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_218.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_218 failure
echo "Running bench2drive_split/bench2drive_218 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=69 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_218.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_218_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_218.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_145 success
echo "Running bench2drive_split/bench2drive_145 success"
DATAGEN=1 TOWN=Town03 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=70 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_145.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_145_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_145.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_145 failure
echo "Running bench2drive_split/bench2drive_145 failure"
DATAGEN=1 TOWN=Town03 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=71 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_145.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_145_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_145.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_197 success
echo "Running bench2drive_split/bench2drive_197 success"
DATAGEN=1 TOWN=Town05 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=72 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_197.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_197_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_197.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_197 failure
echo "Running bench2drive_split/bench2drive_197 failure"
DATAGEN=1 TOWN=Town05 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=73 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_197.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_197_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_197.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_207 success
echo "Running bench2drive_split/bench2drive_207 success"
DATAGEN=1 TOWN=Town02 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=74 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_207.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_207_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_207.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_207 failure
echo "Running bench2drive_split/bench2drive_207 failure"
DATAGEN=1 TOWN=Town02 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=75 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_207.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_207_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_207.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_84 success
echo "Running bench2drive_split/bench2drive_84 success"
DATAGEN=1 TOWN=Town13 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=76 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_84.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_84_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_84.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_84 failure
echo "Running bench2drive_split/bench2drive_84 failure"
DATAGEN=1 TOWN=Town13 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=77 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_84.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_84_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_84.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_39 success
echo "Running bench2drive_split/bench2drive_39 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=78 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_39.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_39_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_39.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_39 failure
echo "Running bench2drive_split/bench2drive_39 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=79 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_39.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_39_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_39.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_196 success
echo "Running bench2drive_split/bench2drive_196 success"
DATAGEN=1 TOWN=Town05 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=80 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_196.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_196_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_196.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_196 failure
echo "Running bench2drive_split/bench2drive_196 failure"
DATAGEN=1 TOWN=Town05 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=81 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_196.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_196_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_196.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_186 success
echo "Running bench2drive_split/bench2drive_186 success"
DATAGEN=1 TOWN=Town06 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=82 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_186.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_186_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_186.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_186 failure
echo "Running bench2drive_split/bench2drive_186 failure"
DATAGEN=1 TOWN=Town06 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=83 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_186.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_186_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_186.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_03 success
echo "Running bench2drive_split/bench2drive_03 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=84 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_03.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_03_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_03.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_03 failure
echo "Running bench2drive_split/bench2drive_03 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=85 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_03.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_03_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_03.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_20 success
echo "Running bench2drive_split/bench2drive_20 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=86 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_20.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_20_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_20.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_20 failure
echo "Running bench2drive_split/bench2drive_20 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=87 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_20.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_20_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_20.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_212 success
echo "Running bench2drive_split/bench2drive_212 success"
DATAGEN=1 TOWN=Town10HD REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=88 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_212.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_212_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_212.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_212 failure
echo "Running bench2drive_split/bench2drive_212 failure"
DATAGEN=1 TOWN=Town10HD REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=89 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_212.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_212_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_212.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_150 success
echo "Running bench2drive_split/bench2drive_150 success"
DATAGEN=1 TOWN=Town07 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=90 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_150.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_150_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_150.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_150 failure
echo "Running bench2drive_split/bench2drive_150 failure"
DATAGEN=1 TOWN=Town07 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=91 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_150.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_150_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_150.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_45 success
echo "Running bench2drive_split/bench2drive_45 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=92 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_45.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_45_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_45.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_45 failure
echo "Running bench2drive_split/bench2drive_45 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=93 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_45.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_45_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_45.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_08 success
echo "Running bench2drive_split/bench2drive_08 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=94 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_08.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_08_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_08.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_08 failure
echo "Running bench2drive_split/bench2drive_08 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=95 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_08.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_08_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_08.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_02 success
echo "Running bench2drive_split/bench2drive_02 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=96 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_02.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_02_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_02.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_02 failure
echo "Running bench2drive_split/bench2drive_02 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=97 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_02.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_02_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_02.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_219 success
echo "Running bench2drive_split/bench2drive_219 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=98 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_219.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_219_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_219.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_219 failure
echo "Running bench2drive_split/bench2drive_219 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=99 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_219.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_219_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_219.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_180 success
echo "Running bench2drive_split/bench2drive_180 success"
DATAGEN=1 TOWN=Town05 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=100 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_180.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_180_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_180.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_180 failure
echo "Running bench2drive_split/bench2drive_180 failure"
DATAGEN=1 TOWN=Town05 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=101 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_180.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_180_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_180.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_125 success
echo "Running bench2drive_split/bench2drive_125 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=102 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_125.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_125_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_125.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_125 failure
echo "Running bench2drive_split/bench2drive_125 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=103 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_125.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_125_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_125.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_24 success
echo "Running bench2drive_split/bench2drive_24 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=104 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_24.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_24_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_24.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_24 failure
echo "Running bench2drive_split/bench2drive_24 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=105 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_24.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_24_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_24.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_208 success
echo "Running bench2drive_split/bench2drive_208 success"
DATAGEN=1 TOWN=Town04 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=106 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_208.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_208_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_208.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_208 failure
echo "Running bench2drive_split/bench2drive_208 failure"
DATAGEN=1 TOWN=Town04 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=107 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_208.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_208_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_208.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_52 success
echo "Running bench2drive_split/bench2drive_52 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=108 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_52.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_52_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_52.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_52 failure
echo "Running bench2drive_split/bench2drive_52 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=109 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_52.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_52_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_52.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_38 success
echo "Running bench2drive_split/bench2drive_38 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=110 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_38.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_38_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_38.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_38 failure
echo "Running bench2drive_split/bench2drive_38 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=111 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_38.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_38_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_38.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_72 success
echo "Running bench2drive_split/bench2drive_72 success"
DATAGEN=1 TOWN=Town13 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=112 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_72.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_72_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_72.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_72 failure
echo "Running bench2drive_split/bench2drive_72 failure"
DATAGEN=1 TOWN=Town13 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=113 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_72.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_72_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_72.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_157 success
echo "Running bench2drive_split/bench2drive_157 success"
DATAGEN=1 TOWN=Town05 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=114 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_157.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_157_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_157.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_157 failure
echo "Running bench2drive_split/bench2drive_157 failure"
DATAGEN=1 TOWN=Town05 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=115 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_157.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_157_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_157.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_50 success
echo "Running bench2drive_split/bench2drive_50 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=116 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_50.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_50_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_50.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_50 failure
echo "Running bench2drive_split/bench2drive_50 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=117 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_50.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_50_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_50.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_147 success
echo "Running bench2drive_split/bench2drive_147 success"
DATAGEN=1 TOWN=Town02 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=118 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_147.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_147_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_147.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_147 failure
echo "Running bench2drive_split/bench2drive_147 failure"
DATAGEN=1 TOWN=Town02 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=119 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_147.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_147_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_147.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_177 success
echo "Running bench2drive_split/bench2drive_177 success"
DATAGEN=1 TOWN=Town03 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=120 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_177.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_177_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_177.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_177 failure
echo "Running bench2drive_split/bench2drive_177 failure"
DATAGEN=1 TOWN=Town03 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=121 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_177.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_177_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_177.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_153 success
echo "Running bench2drive_split/bench2drive_153 success"
DATAGEN=1 TOWN=Town15 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=122 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_153.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_153_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_153.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_153 failure
echo "Running bench2drive_split/bench2drive_153 failure"
DATAGEN=1 TOWN=Town15 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=123 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_153.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_153_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_153.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_58 success
echo "Running bench2drive_split/bench2drive_58 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=124 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_58.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_58_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_58.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_58 failure
echo "Running bench2drive_split/bench2drive_58 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=125 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_58.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_58_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_58.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_166 success
echo "Running bench2drive_split/bench2drive_166 success"
DATAGEN=1 TOWN=Town06 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=126 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_166.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_166_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_166.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_166 failure
echo "Running bench2drive_split/bench2drive_166 failure"
DATAGEN=1 TOWN=Town06 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=127 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_166.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_166_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_166.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_216 success
echo "Running bench2drive_split/bench2drive_216 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=128 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_216.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_216_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_216.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_216 failure
echo "Running bench2drive_split/bench2drive_216 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=129 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_216.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_216_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_216.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_210 success
echo "Running bench2drive_split/bench2drive_210 success"
DATAGEN=1 TOWN=Town04 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=130 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_210.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_210_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_210.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_210 failure
echo "Running bench2drive_split/bench2drive_210 failure"
DATAGEN=1 TOWN=Town04 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=131 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_210.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_210_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_210.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_191 success
echo "Running bench2drive_split/bench2drive_191 success"
DATAGEN=1 TOWN=Town03 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=132 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_191.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_191_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_191.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_191 failure
echo "Running bench2drive_split/bench2drive_191 failure"
DATAGEN=1 TOWN=Town03 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=133 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_191.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_191_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_191.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_168 success
echo "Running bench2drive_split/bench2drive_168 success"
DATAGEN=1 TOWN=Town05 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=134 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_168.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_168_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_168.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_168 failure
echo "Running bench2drive_split/bench2drive_168 failure"
DATAGEN=1 TOWN=Town05 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=135 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_168.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_168_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_168.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_96 success
echo "Running bench2drive_split/bench2drive_96 success"
DATAGEN=1 TOWN=Town13 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=136 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_96.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_96_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_96.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_96 failure
echo "Running bench2drive_split/bench2drive_96 failure"
DATAGEN=1 TOWN=Town13 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=137 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_96.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_96_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_96.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_121 success
echo "Running bench2drive_split/bench2drive_121 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=138 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_121.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_121_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_121.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_121 failure
echo "Running bench2drive_split/bench2drive_121 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=139 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_121.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_121_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_121.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_179 success
echo "Running bench2drive_split/bench2drive_179 success"
DATAGEN=1 TOWN=Town04 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=140 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_179.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_179_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_179.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_179 failure
echo "Running bench2drive_split/bench2drive_179 failure"
DATAGEN=1 TOWN=Town04 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=141 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_179.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_179_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_179.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_41 success
echo "Running bench2drive_split/bench2drive_41 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=142 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_41.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_41_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_41.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_41 failure
echo "Running bench2drive_split/bench2drive_41 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=143 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_41.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_41_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_41.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_122 success
echo "Running bench2drive_split/bench2drive_122 success"
DATAGEN=1 TOWN=Town12 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=144 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_122.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_122_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_122.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_122 failure
echo "Running bench2drive_split/bench2drive_122 failure"
DATAGEN=1 TOWN=Town12 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=145 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_122.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_122_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_122.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_164 success
echo "Running bench2drive_split/bench2drive_164 success"
DATAGEN=1 TOWN=Town04 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=146 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_164.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_164_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_164.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_164 failure
echo "Running bench2drive_split/bench2drive_164 failure"
DATAGEN=1 TOWN=Town04 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=147 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_164.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_164_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_164.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_198 success
echo "Running bench2drive_split/bench2drive_198 success"
DATAGEN=1 TOWN=Town03 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=148 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_198.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_198_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_198.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_198 failure
echo "Running bench2drive_split/bench2drive_198 failure"
DATAGEN=1 TOWN=Town03 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=149 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_198.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_198_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_198.xml --debug=0 --resume=1 --timeout=600
after_datagen_run

# bench2drive_split/bench2drive_82 success
echo "Running bench2drive_split/bench2drive_82 success"
DATAGEN=1 TOWN=Town13 REPETITION=0 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/success SIM_CASE_LABEL=success SIM_FAILURE_DISTURB=0 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=0 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=150 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_82.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/success/bench2drive_82_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_82.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
# bench2drive_split/bench2drive_82 failure
echo "Running bench2drive_split/bench2drive_82 failure"
DATAGEN=1 TOWN=Town13 REPETITION=1 SAVE_PATH=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/bench2drive_split/failure SIM_CASE_LABEL=failure SIM_FAILURE_DISTURB=1 SIM_FAILURE_STOP_AFTER_COLLISION=1 SIM_FAILURE_COLLISION_TAIL_SECONDS=10 "${PYTHON_BIN}" "${WORK_DIR}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py" --port=${FREE_WORLD_PORT:-2000} --traffic-manager-port=${TM_PORT:-8000} --traffic-manager-seed=151 --routes=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_82.xml --repetitions=1 --track=MAP --checkpoint=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/new_training_data/results/bench2drive_split/failure/bench2drive_82_result.json --agent="${WORK_DIR}/team_code/data_agent.py" --agent-config=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/leaderboard/data/bench2drive_split/bench2drive_82.xml --debug=0 --resume=1 --timeout=600
after_datagen_run
