#!/usr/bin/env python3
"""
merge_floor_graphs.py

Combine two node/edge JSON files into one, then:
  • find the *lowest* (i.e. greatest y-value) elevator in file-2
  • find the first node with "full":true in file-1
  • add an undirected edge between those two IDs
  • write merged JSON
  • optionally display a plot

USAGE
-----
    python merge_floor_graphs.py floor10_partA.json floor10_partB.json \
        -o floor10_merged.json --plot
"""
import argparse, json, pathlib, sys
import matplotlib.pyplot as plt

def load(fp):
    with open(fp, "r") as f:
        return json.load(f)

def index(nodes):
    return {n["id"]: n for n in nodes}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("file1", help="first JSON (contains a 'full:true' node)")
    ap.add_argument("file2", help="second JSON (contains elevator nodes)")
    ap.add_argument("-o", "--out", default="merged.json",
                    help="output filename (default: merged.json)")
    ap.add_argument("--plot", action="store_true",
                    help="show a quick scatter/edge plot")
    args = ap.parse_args(argv)

    d1, d2 = load(args.file1), load(args.file2)

    # ❶ merge node dictionaries (keep first file's tags if duplicate IDs)
    nodes = index(d1["nodes"])
    nodes.update(index(d2["nodes"]))              # second file overrides duplicates

    # ❷ gather edges into a set of unordered tuples to avoid duplicates
    edges = {tuple(e) for e in d1.get("edges", []) + d2.get("edges", [])}

    # ❸ find special nodes
    full_node = next((n for n in d1["nodes"] if n["tags"].get("full")), None)
    elev_nodes = [n for n in d2["nodes"] if n["tags"].get("elevator")]
    elev_node  = max(elev_nodes, key=lambda n: n["y"]) if elev_nodes else None

    if not full_node or not elev_node:
        sys.exit("ERROR: could not find the required 'full' or 'elevator' node.")

    # ❹ add connecting edge
    edges.add(tuple((full_node["id"], elev_node["id"])))

    # ❺ write merged JSON
    merged = {"nodes": list(nodes.values()),
              "edges": [list(e) for e in edges]}
    pathlib.Path(args.out).write_text(json.dumps(merged, indent=2))
    print(f"✓ merged graph written to {args.out}")

    # ❻ optional plot
    if args.plot:
        id2pt = {n["id"]: (n["x"], n["y"]) for n in merged["nodes"]}
        xs, ys = zip(*id2pt.values())
        plt.figure(figsize=(10, 6))
        plt.scatter(xs, ys, s=10, alpha=0.8, label="nodes")

        for u, v in merged["edges"]:
            x0, y0 = id2pt[u]
            x1, y1 = id2pt[v]
            plt.plot([x0, x1], [y0, y1], linewidth=0.8, alpha=0.6)

        # highlight the special pair
        plt.scatter([full_node["x"]], [full_node["y"]],
                    s=50, c="red", label="full:true node")
        plt.scatter([elev_node["x"]], [elev_node["y"]],
                    s=50, c="green", label="elevator (max-y)")

        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.legend()
        plt.title("Merged Floor-10 Graph")
        plt.show()

if __name__ == "__main__":
    main()
