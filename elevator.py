#!/usr/bin/env python3
"""
refactor_elevators_gui.py – interactive elevator alignment preview (elevators only)

Usage:
    python refactor_elevators_gui.py input.json output.json

Features:
  • Load JSON nodes/edges
  • Only elevator nodes are drawn
  • Slider to adjust clustering threshold
  • Preview: original elevators in red; aligned positions in green
  • Confirm with 'c' to apply and save, or 'q' to exit without saving

Controls:
  • Trackbar 'Threshold' to change distance threshold
  • 'c' key: confirm and write output JSON
  • 'q' key: quit without saving

Note:
  Alignment uses the first elevator node in each cluster as reference. Moves each elevator's entire connected component accordingly.
"""
import os
# Force Qt to use XCB plugin instead of Wayland
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

import cv2, json, argparse, math
from pathlib import Path
from collections import deque
import numpy as np

def euclidean(a, b):
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])

def build_adjacency(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return adj

def bfs_component(start, adj):
    seen = {start}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return seen

# clustering function
def cluster_elevators(nodes, threshold):
    elevs = [n for n in nodes if n.get('tags', {}).get('elevator')]
    clusters = []
    for n in elevs:
        placed = False
        for c in clusters:
            if euclidean(n, c[0]) <= threshold:
                c.append(n)
                placed = True
                break
        if not placed:
            clusters.append([n])
    return clusters

# visualization colors and radius
COL_ORIG = (0, 0, 255)   # original elevator nodes in red
COL_NEW  = (0, 200, 0)   # aligned elevator positions in green
RADIUS   = 6             # node radius in preview

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Align elevator clusters interactively")
    parser.add_argument('input', help='Input JSON file')
    parser.add_argument('output', help='Output JSON file')
    args = parser.parse_args()

    # Load data
    data  = json.loads(Path(args.input).read_text())
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])
    adj   = build_adjacency(edges)

    # Filter elevator nodes
    elevators = [n for n in nodes if n.get('tags', {}).get('elevator')]
    if not elevators:
        print("No elevator-tagged nodes found.")
        exit(1)

    # Compute bounds for scaling
    xs, ys = [n['x'] for n in elevators], [n['y'] for n in elevators]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    W = 800
    H = int((maxy - miny) / (maxx - minx) * W) if maxx > minx else 600
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    # Setup preview window and slider
    def nothing(x): pass
    cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Threshold', 'preview', 5, 2000, nothing)

    confirmed = False
    while True:
        thresh = float(cv2.getTrackbarPos('Threshold', 'preview'))
        img = canvas.copy()

        # Draw original elevator positions
        for n in elevators:
            x = int((n['x'] - minx) / (maxx - minx) * (W - 20) + 10)
            y = int((n['y'] - miny) / (maxy - miny) * (H - 20) + 10)
            cv2.circle(img, (x, y), RADIUS, COL_ORIG, -1)

        # Cluster and preview: show new elevator positions only
        clusters = cluster_elevators(nodes, thresh)
        for c in clusters:
            avgx = sum(n['x'] for n in c) / len(c)
            avgy = sum(n['y'] for n in c) / len(c)
            nx = int((avgx - minx) / (maxx - minx) * (W - 20) + 10)
            ny = int((avgy - miny) / (maxy - miny) * (H - 20) + 10)
            cv2.circle(img, (nx, ny), RADIUS, COL_NEW, -1)

        cv2.imshow('preview', img)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('c'):
            # Apply alignment: for each elevator, move its entire component
            for c in clusters:
                avgx = sum(n['x'] for n in c) / len(c)
                avgy = sum(n['y'] for n in c) / len(c)
                for n in c:
                    dx = avgx - n['x']
                    dy = avgy - n['y']
                    comp = bfs_component(n['id'], adj)
                    for nid in comp:
                        node = next(x for x in nodes if x['id'] == nid)
                        node['x'] += dx
                        node['y'] += dy
            # Save
            Path(args.output).write_text(json.dumps(data, indent=2))
            print(f"Saved aligned JSON to {args.output}")
            confirmed = True
            break
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    if not confirmed:
        print("Exited without saving.")
