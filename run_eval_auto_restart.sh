#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARLA_DIR="${SCRIPT_DIR}/carla_garage/carla"
LOG_DIR="${SCRIPT_DIR}/logs"
CARLA_STARTUP_WAIT="${CARLA_STARTUP_WAIT:-20}"
CARLA_RESTART_WAIT="${CARLA_RESTART_WAIT:-5}"
RESTART_EVAL_ON_FAILURE="${RESTART_EVAL_ON_FAILURE:-1}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"

mkdir -p "${LOG_DIR}"

CARLA_PID=""
EVAL_PID=""
STOP_REQUESTED=0
RESTART_COUNT=0

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $*"
}

start_carla() {
    local log_file="${LOG_DIR}/carla_$(date +%Y%m%d_%H%M%S).log"

    log "Starting CARLA server. Log: ${log_file}"
    setsid bash -c 'cd "$1" || exit 1; exec ./CarlaUE4.sh -RenderOffScreen' _ "${CARLA_DIR}" >"${log_file}" 2>&1 &

    CARLA_PID=$!
    log "CARLA PID: ${CARLA_PID}"
}

stop_carla() {
    if [[ -n "${CARLA_PID}" ]] && kill -0 "${CARLA_PID}" 2>/dev/null; then
        log "Stopping CARLA PID ${CARLA_PID}"
        kill -9 "${CARLA_PID}"
    fi
}

start_eval() {
    log "Starting evaluation"
    (
        cd "${SCRIPT_DIR}" || exit 1
        ./run_eval.sh
    ) &
    EVAL_PID=$!
    log "Evaluation PID: ${EVAL_PID}"
}

stop_eval() {
    if [[ -n "${EVAL_PID}" ]] && kill -0 "${EVAL_PID}" 2>/dev/null; then
        log "Stopping evaluation PID ${EVAL_PID}"
        kill -9 "${EVAL_PID}"
    fi
}

cleanup() {
    STOP_REQUESTED=1
    stop_eval
    stop_carla
}

handle_interrupt() {
    cleanup
    exit 130
}

trap handle_interrupt INT TERM

if [[ ! -x "${CARLA_DIR}/CarlaUE4.sh" ]]; then
    log "ERROR: Cannot execute ${CARLA_DIR}/CarlaUE4.sh"
    exit 1
fi

while true; do
    start_carla
    log "Waiting ${CARLA_STARTUP_WAIT}s before starting evaluation"
    sleep "${CARLA_STARTUP_WAIT}"
    start_eval

    EVAL_STATUS=1
    RESTART_REASON=""

    while true; do
        if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
            wait "${CARLA_PID}"
            EVAL_STATUS=$?
            log "Evaluation finished with exit code ${EVAL_STATUS}"
            stop_carla
            sleep 15
            stop_eval
            sleep 15
            break
        fi

        if [[ "${EVAL_STATUS}" -eq 0 ]]; then
                exit 0
        fi
        sleep 10
    done

    if [[ "${STOP_REQUESTED}" -eq 1 ]]; then
        exit 130
    fi

    if [[ "${RESTART_REASON}" == "evaluation exited with failure" && "${RESTART_EVAL_ON_FAILURE}" != "1" ]]; then
        exit "${EVAL_STATUS}"
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    if [[ "${MAX_RESTARTS}" -gt 0 && "${RESTART_COUNT}" -gt "${MAX_RESTARTS}" ]]; then
        log "Maximum restart count reached (${MAX_RESTARTS}); exiting"
        exit "${EVAL_STATUS}"
    fi

    log "Restarting workflow because ${RESTART_REASON}. Restart #${RESTART_COUNT} in ${CARLA_RESTART_WAIT}s"
    sleep "${CARLA_RESTART_WAIT}"
done
