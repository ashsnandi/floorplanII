#!/usr/bin/env python3
"""
plot_nodes.py
-------------
Quick-n-dirty visualizer for node JSON.

Usage
-----
    python plot_nodes.py path/to/floorplan.json [--annotate] [--save PNGFILE]

The JSON schema it expects:

{
  "nodes": [
     {"id": "J2502_1238", "x": 2502.0, "y": 1238.0, "floor": 0},
     ...
  ],  // OPTIONAL – if present, will be drawn as straight lines
  "edges": [
     {"u": "J2502_1238", "v": "J2503_1238"},
     ...
  ]
}
"""
import argparse, json, math, pathlib, sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def load(fp: pathlib.Path):
    with fp.open() as f:
        data = json.load(f)
    nodes = data["nodes"]
    edges = data.get("edges", [])        # optional
    return nodes, edges

def make_axes(n_floors):
    # 1×k or ⌈sqrt⌉ grid depending on count
    if n_floors == 1:
        fig, ax = plt.subplots(figsize=(8, 8))
        return fig, [ax]
    r = math.ceil(math.sqrt(n_floors))
    c = math.ceil(n_floors / r)
    fig, axs = plt.subplots(r, c, figsize=(4 * c, 4 * r))
    return fig, axs.flat

def plot_floor(ax, nodes, edges, annotate=False):
    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]
    ax.scatter(xs, ys, s=12, zorder=2)

    if annotate:
        for n in nodes:
            ax.text(n["x"], n["y"], n["id"], fontsize=6, ha="left", va="bottom")

    # Draw edges (straight-line preview) if any
    if edges:
        id2pt = {n["id"]: (n["x"], n["y"]) for n in nodes}
        for e in edges:
            if e["u"] in id2pt and e["v"] in id2pt:
                x0, y0 = id2pt[e["u"]]
                x1, y1 = id2pt[e["v"]]
                ax.plot([x0, x1], [y0, y1], linewidth=0.8, color="black", zorder=1)

    ax.set_aspect("equal")
    ax.axis("off")

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("json_file", type=pathlib.Path, help="nodes/edges JSON file")
    p.add_argument("--annotate", action="store_true",
                   help="draw small text labels for node IDs")
    p.add_argument("--save", metavar="PNGFILE",
                   help="save to PNG instead of showing an interactive window")
    args = p.parse_args(argv)

    nodes, edges = load(args.json_file)

    # Split by floor
    by_floor = defaultdict(list)
    for n in nodes:
        by_floor[n["floor"]].append(n)

    fig, axes = make_axes(len(by_floor))
    for ax, (floor, f_nodes) in zip(axes, sorted(by_floor.items())):
        ax.set_title(f"Floor {floor}", fontsize=10)
        plot_floor(ax, f_nodes, edges, annotate=args.annotate)

    # Hide any leftover empty axes
    for ax in axes[len(by_floor):]:
        ax.axis("off")

    fig.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=300)
        print(f"Saved to {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
