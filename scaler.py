#!/usr/bin/env python3
"""
rescale_nodes.py  –  affine-transform an editor JSON to master coordinates.

Usage
-----
python rescale_nodes.py floorX_raw.json \
        --src  x1 y1  x2 y2            # ref pts in floor X image
        --dst  X1 Y1  X2 Y2            # matching pts in master image
        -o floorX_scaled.json

For three reference points, repeat --src/--dst once more.
"""
import argparse, json, numpy as np, pathlib, sys

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("infile")
    p.add_argument("--src", type=float, nargs="+", required=True,
                   help="pixel coords in the CURRENT image (x1 y1 x2 y2 [...])")
    p.add_argument("--dst", type=float, nargs="+", required=True,
                   help="matching coords in the MASTER image")
    p.add_argument("-o", "--out", default="scaled.json")
    return p.parse_args()

def solve_affine(src, dst):
    src, dst = np.array(src).reshape(-1,2), np.array(dst).reshape(-1,2)
    if len(src) != len(dst) or len(src) < 2:
        sys.exit("Need ≥2 matching points.")
    # build linear system  [x y 1 0 0 0]*[a b tx]^T = X
    A = []
    B = []
    for (x,y),(X,Y) in zip(src,dst):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.extend([X, Y])
    A = np.array(A); B = np.array(B)
    params, *_ = np.linalg.lstsq(A, B, rcond=None)
    a,b,tx,c,d,ty = params            # affine matrix [[a b tx],[c d ty]]
    return np.array([[a,b,tx],[c,d,ty],[0,0,1]])

def main():
    args = parse()
    M = solve_affine(args.src, args.dst)          # 3×3
    data = json.loads(pathlib.Path(args.infile).read_text())

    for n in data["nodes"]:
        v = np.array([n["x"], n["y"], 1.0])
        X,Y,_ = M @ v
        n["x"], n["y"] = float(X), float(Y)

    pathlib.Path(args.out).write_text(json.dumps(data, indent=2))
    print("→ wrote", args.out)

if __name__ == "__main__":
    main()
