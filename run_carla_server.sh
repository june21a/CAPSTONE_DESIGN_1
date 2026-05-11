#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CARLA_DIR="${SCRIPT_DIR}/carla_garage/carla"
echo "${CARLA_DIR}"

cd "${CARLA_DIR}" || exit
./CarlaUE4.sh -RenderOffScreen