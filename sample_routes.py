#!/usr/bin/env python3
"""Randomly sample route definitions from a CARLA leaderboard XML file."""

from __future__ import annotations

import argparse
import copy
import random
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_INPUT = Path("carla_garage/leaderboard/data/bench2drive220.xml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a smaller route XML by randomly sampling <route> elements "
            "from a full evaluation route file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source route XML. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output XML path. Default: next to the input file as "
            "<input_stem>_sample<num>_seed<seed>.xml"
        ),
    )
    parser.add_argument(
        "-n",
        "--num-routes",
        type=int,
        default=55,
        help="Number of routes to sample. Default: 20",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible route samples. Default: 0",
    )
    parser.add_argument(
        "--town",
        action="append",
        default=None,
        help=(
            "Only sample routes from this town. Can be passed multiple times, "
            "for example: --town Town12 --town Town13"
        ),
    )
    parser.add_argument(
        "--keep-random-order",
        action="store_true",
        help="Keep the random draw order. By default, sampled routes keep their original XML order.",
    )
    parser.add_argument(
        "--print-ids",
        action="store_true",
        help="Print sampled route IDs as a comma-separated list.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path, num_routes: int, seed: int) -> Path:
    return input_path.with_name(f"{input_path.stem}_sample{num_routes}_seed{seed}{input_path.suffix}")


def read_routes(input_path: Path) -> tuple[ET.ElementTree, ET.Element, list[ET.Element]]:
    tree = ET.parse(input_path)
    root = tree.getroot()
    if root.tag != "routes":
        raise ValueError(f"Expected root tag <routes>, got <{root.tag}> in {input_path}")

    routes = list(root.findall("route"))
    if not routes:
        raise ValueError(f"No <route> elements found in {input_path}")

    return tree, root, routes


def sample_routes(
    routes: list[ET.Element],
    num_routes: int,
    seed: int,
    towns: set[str] | None = None,
    keep_random_order: bool = False,
) -> list[ET.Element]:
    if towns:
        candidates = [route for route in routes if route.get("town") in towns]
    else:
        candidates = list(routes)

    if num_routes <= 0:
        raise ValueError("--num-routes must be greater than 0")
    if num_routes > len(candidates):
        town_msg = f" for towns {sorted(towns)}" if towns else ""
        raise ValueError(
            f"Requested {num_routes} routes, but only {len(candidates)} are available{town_msg}."
        )

    rng = random.Random(seed)
    selected = rng.sample(candidates, num_routes)

    if keep_random_order:
        return selected

    selected_ids = {id(route) for route in selected}
    return [route for route in routes if id(route) in selected_ids]


def write_routes(output_path: Path, sampled_routes: list[ET.Element]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sampled_root = ET.Element("routes")
    for route in sampled_routes:
        sampled_root.append(copy.deepcopy(route))

    ET.indent(sampled_root, space="   ")
    ET.ElementTree(sampled_root).write(output_path, encoding="utf-8", xml_declaration=False)


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or default_output_path(input_path, args.num_routes, args.seed)
    towns = set(args.town) if args.town else None

    _, _, routes = read_routes(input_path)
    sampled = sample_routes(
        routes,
        num_routes=args.num_routes,
        seed=args.seed,
        towns=towns,
        keep_random_order=args.keep_random_order,
    )
    write_routes(output_path, sampled)

    sampled_ids = [route.get("id", "") for route in sampled]
    print(f"Read {len(routes)} routes from {input_path}")
    print(f"Wrote {len(sampled)} sampled routes to {output_path}")
    if args.print_ids:
        print(",".join(sampled_ids))


if __name__ == "__main__":
    main()
