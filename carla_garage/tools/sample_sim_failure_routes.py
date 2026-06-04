#!/usr/bin/env python3
"""Sample CARLA route XMLs for paired success/failure simulator data collection."""

from __future__ import annotations

import argparse
import json
import random
import re
import shlex
import xml.etree.ElementTree as ET
from pathlib import Path


FOCUS_KEYWORDS = (
    "Intersection",
    "Junction",
    "SignalizedJunction",
    "NoSignalJunction",
    "Crossing",
    "CutIn",
    "ChangeLane",
    "LaneChange",
    "ParkingExit",
    "InvadingTurn",
    "AccidentTwoWays",
    "ConstructionObstacleTwoWays",
    "ParkedObstacleTwoWays",
    "HazardAtSideLane",
    "Merger",
)


def route_town(route_file: Path) -> str:
    try:
        root = ET.parse(route_file).getroot()
        route = root.find(".//route")
        if route is not None and route.get("town"):
            return route.get("town")
    except ET.ParseError:
        pass

    match = re.search(r"Town\d+", str(route_file))
    return match.group(0) if match else ""


def is_focus_route(route_file: Path) -> bool:
    scenario_name = route_file.parent.name
    if any(keyword.lower() in scenario_name.lower() for keyword in FOCUS_KEYWORDS):
        return True

    try:
        text = route_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(keyword.lower() in text.lower() for keyword in FOCUS_KEYWORDS)


def make_run_pair(route_file: Path, output_root: Path, carla_garage_root: Path, seed: int) -> dict:
    scenario = route_file.parent.name
    route_id = route_file.stem
    town = route_town(route_file)

    def command(case_label: str, repetition: int, disturb: bool, seed_offset: int) -> str:
        save_path = output_root / scenario / case_label
        checkpoint = output_root / "results" / scenario / case_label / f"{route_id}_result.json"
        env = {
            "DATAGEN": "1",
            "TOWN": town,
            "REPETITION": str(repetition),
            "SAVE_PATH": str(save_path),
            "SIM_CASE_LABEL": case_label,
            "SIM_FAILURE_DISTURB": "1" if disturb else "0",
            "SIM_FAILURE_STOP_AFTER_COLLISION": "1",
            "SIM_FAILURE_COLLISION_TAIL_SECONDS": "10" if disturb else "0",
        }
        exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        return (
            f"{exports} \"${{PYTHON_BIN}}\" \"${{WORK_DIR}}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py\" "
            "--port=${FREE_WORLD_PORT:-2000} "
            "--traffic-manager-port=${TM_PORT:-8000} "
            f"--traffic-manager-seed={seed + seed_offset} "
            f"--routes={shlex.quote(str(route_file))} --repetitions=1 --track=MAP "
            f"--checkpoint={shlex.quote(str(checkpoint))} --agent=\"${{WORK_DIR}}/team_code/data_agent.py\" "
            f"--agent-config={shlex.quote(str(route_file))} --debug=0 --resume=1 --timeout=600"
        )

    return {
        "route_file": str(route_file),
        "scenario": scenario,
        "route_id": route_id,
        "town": town,
        "focus": is_focus_route(route_file),
        "success_command": command("success", 0, False, 0),
        "failure_command": command("failure", 1, True, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample 25% of routes, prioritizing crossroads/lane-change cases.")
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--carla-garage-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--commands",
        type=Path,
        default=None,
        help="Bash file with paired run commands. Defaults to CAPSTONE_DESIGN_1/run_sim_failure_pairs.sh.",
    )
    args = parser.parse_args()
    project_root = args.carla_garage_root.resolve().parent
    if args.commands is None:
        args.commands = project_root / "run_sim_failure_pairs.sh"

    route_files = sorted(args.route_root.glob("**/*.xml"))
    if not route_files:
        raise FileNotFoundError(f"No route XML files under {args.route_root}")

    rng = random.Random(args.seed)
    focus_routes = [route for route in route_files if is_focus_route(route)]
    other_routes = [route for route in route_files if route not in focus_routes]
    rng.shuffle(focus_routes)
    rng.shuffle(other_routes)

    sample_count = max(1, round(len(route_files) * args.fraction))
    sampled = (focus_routes + other_routes)[:sample_count]
    pairs = [make_run_pair(route, args.output_root, args.carla_garage_root, args.seed + idx * 2)
             for idx, route in enumerate(sampled)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "route_root": str(args.route_root),
        "fraction": args.fraction,
        "total_routes": len(route_files),
        "sampled_routes": len(pairs),
        "focus_routes_available": len(focus_routes),
        "pairs": pairs,
    }, indent=2), encoding="utf-8")

    if args.commands is not None:
        args.commands.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "",
            "# ===== CARLA Environment =====",
            "CURRENT_DIR=$(pwd)",
            "export CARLA_ROOT=\"${CURRENT_DIR}/carla_garage/carla\"",
            "export WORK_DIR=\"${CURRENT_DIR}/carla_garage\"",
            "export SCENARIO_RUNNER_ROOT=\"${WORK_DIR}/scenario_runner_autopilot\"",
            "export LEADERBOARD_ROOT=\"${WORK_DIR}/leaderboard_autopilot\"",
            "export PYTHON_BIN=\"${PYTHON_BIN:-python}\"",
            "",
            "# ===== PYTHONPATH =====",
            "export PYTHONPATH=\"${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH:-}\"",
            "export LD_LIBRARY_PATH=\"${CARLA_ROOT}/PythonAPI/carla/dist:${CARLA_ROOT}/LibCarla/lib:${LD_LIBRARY_PATH:-}\"",
            "",
            "# ===== Defaults =====",
            "export FREE_WORLD_PORT=\"${FREE_WORLD_PORT:-2000}\"",
            "export TM_PORT=\"${TM_PORT:-8000}\"",
            "export SIM_FAILURE_COLLISION_TAIL_SECONDS=\"${SIM_FAILURE_COLLISION_TAIL_SECONDS:-10}\"",
            "",
            "cd \"${WORK_DIR}\"",
            "",
        ]
        for pair in pairs:
            lines.append(f"# {pair['scenario']}/{pair['route_id']} success")
            lines.append(f"echo \"Running {pair['scenario']}/{pair['route_id']} success\"")
            lines.append(pair["success_command"])
            lines.append(f"# {pair['scenario']}/{pair['route_id']} failure")
            lines.append(f"echo \"Running {pair['scenario']}/{pair['route_id']} failure\"")
            lines.append(pair["failure_command"])
            lines.append("")
        args.commands.write_text("\n".join(lines), encoding="utf-8")
        args.commands.chmod(0o755)

    print(f"Sampled {len(pairs)}/{len(route_files)} routes")
    print(f"Wrote {args.output}")
    if args.commands is not None:
        print(f"Wrote {args.commands}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
