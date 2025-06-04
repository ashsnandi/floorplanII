#!/usr/bin/env python3
"""
floorplan_editor.py — interactive multi‐floor floorplan node/edge editor

Features:
  • Multi‐floor support: switch with [ / ]
  • Per‐floor background images (loaded via --image floor:path)
  • Scaling of each floor image with + / –
  • Modes: normal, hallway, branch, edge, tag, save, quit
  • Tagging: hallway / elevator / full (1/2/3), plus wing (A–Z) & binType (1=N,2=L,3=B,4=M)
  • Auto‐chaining of hallway nodes; branch‐mode auto‐links to nearest hallway
  • Zoom (scale factor) applied only to the background, node coords stay in “world” space
  • Save/load JSON with per‐node attrs: id, x, y, floor, tags, wing, binType
  • Edges survive save/load; each floor stores its own nodes/edges
Controls:
  n: normal (create node at click)
  h: hallway (auto‐chain)
  c: continue (alias for hallway)
  b: branch (auto‐link to nearest hallway)
  e: edge (click two nodes)
  t: tag mode (click node, then press 1/2/3 for tags, A–Z for wing, 1–4 for binType)
  [: switch to previous floor
  ]: switch to next floor
  +: increase current floor’s image scale
  -: decrease current floor’s image scale
  s: save JSON
  q: quit
"""

import cv2
import json
import pathlib
import argparse
import numpy as np
import tkinter as tk
# -------------- Appearance constants --------------
NODE_RADIUS  = 8
CLICK_THRESH = 15
EDGE_COLOR   = (50, 150, 255)
EDGE_THICK   = 3

# Colors for node circle + outline
NODE_FILL    = (0, 255, 255)
NODE_OUTLINE = (0,   0,   0)
SEL_OUTLINE  = (0,   0, 255)

# Colors for tags: we draw small text above/below the node
TAG_COLOR    = (0,   0, 255)   # for hallway/elevator/full badge
WING_COLOR   = (0, 255,   0)   # for wing label
BIN_COLOR    = (255, 0,   0)   # for binType label

# -------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Interactive multi‐floor floorplan editor")
    p.add_argument(
        "--image",
        action="append",
        default=[],
        help="specify floor:image_path, e.g. 0:floor0.png; can repeat for each floor",
    )
    p.add_argument(
        "--json", required=True, help="path to nodes/edges JSON file"
    )
    p.add_argument(
        "--scale", type=float, default=1.0, help="initial global UI zoom factor"
    )
    return p.parse_args()


def main():
    args = parse_args()
    json_path = pathlib.Path(args.json)
    init_scale = args.scale

    # 1. Parse the --image FLOOR:PATH arguments into a dict
    #    images_by_floor[floor] = original CV2 image (unscaled)
    images_by_floor: dict[int, dict] = {}  # floor -> {"orig":img, "disp":disp, "scale":float}
    for spec in args.image:
        try:
            floor_str, img_path = spec.split(":", 1)
            fl = int(floor_str)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: could not load image at {img_path} for floor {fl}")
                continue
            h, w = img.shape[:2]
            images_by_floor[fl] = {
                "orig": img,
                "scale": init_scale,
                "w": w,
                "h": h,
                "disp": None,   # will be computed
                "dx": 0,        # pan offset X (pixels)
                "dy": 0,        # pan offset Y
            }
        except Exception as e:
            print(f"Invalid --image spec '{spec}': {e}")

    # 2. Load or initialize the JSON data structure
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
        except Exception as e:
            print(f"Error reading JSON: {e}")
            return
    else:
        # Initialize an empty multi‐floor structure:
        data = {
            "nodes": [],   # list of {id,x,y,floor,tags:{…}, wing:str or None, binType:str}
            "edges": [],   # list of [u, v, floor]
        }

    # Ensure fields exist
    data.setdefault("nodes", [])
    data.setdefault("edges", [])

    # 3. Precompute a dictionary floor→nodes/edges subsets on load
    #    (Nodes have a "floor" key; edges assumed to belong to that floor)
    #    We'll redraw from “data” each frame, filtering by current_floor.

    # 4. Track the current floor (default 0)
    current_floor = 0
    # 5. UI mode:
    mode = "normal"
    # 6. For edge‐mode: temporary store of first‐selected node ID
    edge_sel: list[str] = []
    # 7. For chain (hallway) continuation:
    last_hallway: str | None = None
    # 8. For tag mode: which node is selected for tagging?
    tag_target: str | None = None

    # ------------------------------------------------------
    # Helper functions for coordinate conversion & neighbors
    # ------------------------------------------------------
    def get_scaled_display_img(floor: int):
        """
        Returns the display image (pan+scaled) for a given floor.
        Updates images_by_floor[floor]["disp"] accordingly.
        """
        info = images_by_floor.get(floor)
        if not info:
            return None
        orig = info["orig"]
        s = info["scale"]
        h, w = info["h"], info["w"]
        # scale
        disp = cv2.resize(orig, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
        # pan
        dx, dy = int(info["dx"]), int(info["dy"])
        canvas = cv2.copyMakeBorder(
            disp,
            top=max(0, dy),
            bottom=max(0, -dy),
            left=max(0, dx),
            right=max(0, -dx),
            borderType=cv2.BORDER_CONSTANT,
            value=(200, 200, 200),
        )
        # If panned negatively, crop:
        h2, w2 = canvas.shape[:2]
        x0 = max(0, -dx)
        y0 = max(0, -dy)
        canvas = canvas[y0 : y0 + int(h * s), x0 : x0 + int(w * s)]
        info["disp"] = canvas
        return canvas

    def nodes_on_floor():
        return [n for n in data["nodes"] if n.get("floor", 0) == current_floor]

    def edges_on_floor():
        return [e for e in data["edges"] if e[2] == current_floor]

    def id2pt_scaled():
        """
        Build a dict mapping node IDs → (x†scale + panX, y†scale + panY)
        so we can draw them on the display image. 
        """
        pts = {}
        info = images_by_floor.get(current_floor)
        s = info["scale"] if info else init_scale
        dx = info["dx"] if info else 0
        dy = info["dy"] if info else 0
        for n in nodes_on_floor():
            xw, yw = n["x"], n["y"]
            xd = int(xw * s + dx)
            yd = int(yw * s + dy)
            pts[n["id"]] = (xd, yd)
        return pts

    def nearest_node(pt_disp):
        """
        Given a point in display coords, return the nearest node ON THIS FLOOR within threshold.
        """
        pts = id2pt_scaled()
        for n in nodes_on_floor():
            x_disp, y_disp = pts[n["id"]]
            dx = x_disp - pt_disp[0]
            dy = y_disp - pt_disp[1]
            if dx * dx + dy * dy <= CLICK_THRESH * CLICK_THRESH:
                return n
        return None

    def nearest_hallway_node(pt_disp):
        """
        Among all nodes on this floor with tags['hallway']==True, 
        return the one whose display‐distance to pt_disp is minimal.
        """
        pts = id2pt_scaled()
        best, bd = None, float("inf")
        for n in nodes_on_floor():
            if n.get("tags", {}).get("hallway"):
                x0, y0 = pts[n["id"]]
                d = (x0 - pt_disp[0]) ** 2 + (y0 - pt_disp[1]) ** 2
                if d < bd:
                    best, bd = n, d
        return best

    # --------------------------------
    # Redraw function (draw everything)
    # --------------------------------
    canvas = None  # will hold the current display frame

    def redraw():
        nonlocal canvas
        # 1. Start with the background (if exists), else gray canvas
        disp_bg = get_scaled_display_img(current_floor)
        if disp_bg is None:
            # blank gray if no image for this floor
            h = int(600)
            w = int(800)
            canvas = 50 * np.ones((h, w, 3), dtype=np.uint8)
        else:
            canvas = disp_bg.copy()

        # 2. Draw edges on this floor
        pts = id2pt_scaled()
        for u, v, fl in edges_on_floor():
            if u in pts and v in pts:
                cv2.line(canvas, pts[u], pts[v], EDGE_COLOR, EDGE_THICK)

        # 3. Draw nodes on this floor
        for n in nodes_on_floor():
            x_disp, y_disp = pts[n["id"]]
            # fill circle
            cv2.circle(canvas, (x_disp, y_disp), NODE_RADIUS, NODE_FILL, -1)
            # outline
            cv2.circle(canvas, (x_disp, y_disp), NODE_RADIUS, NODE_OUTLINE, 1)
            # If it’s the selected target in edge-mode, highlight
            if mode == "edge" and n["id"] in edge_sel:
                cv2.circle(canvas, (x_disp, y_disp), NODE_RADIUS + 4, SEL_OUTLINE, 2)
            # If it’s the tag target, outline
            if mode == "tag" and n["id"] == tag_target:
                cv2.circle(canvas, (x_disp, y_disp), NODE_RADIUS + 6, SEL_OUTLINE, 2)

            # 3a. Draw tag badges: “H” if hallway, “E” if elevator, “F” if full
            tags = n.get("tags", {})
            badge = ""
            if tags.get("hallway"):
                badge = "H"
            elif tags.get("elevator"):
                badge = "E"
            elif tags.get("full"):
                badge = "F"
            if badge:
                cv2.putText(
                    canvas,
                    badge,
                    (x_disp + NODE_RADIUS + 2, y_disp - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    TAG_COLOR,
                    1,
                )

            # 3b. Draw wing label (above tag badge) if present
            wing = n.get("wing")
            if wing:
                cv2.putText(
                    canvas,
                    wing.upper(),
                    (x_disp - 8, y_disp - NODE_RADIUS - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    WING_COLOR,
                    1,
                )

            # 3c. Draw binType label below node
            binType = n.get("binType", "N")
            if binType:
                cv2.putText(
                    canvas,
                    binType,
                    (x_disp - 6, y_disp + NODE_RADIUS + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    BIN_COLOR,
                    1,
                )

        # 4. Draw current floor text
        cv2.putText(
            canvas,
            f"Floor: {current_floor}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # -----------------------------------
    # Mouse‐click callback for cv2 window
    # -----------------------------------
    def on_click(event, x_disp, y_disp, flags, param):
        nonlocal last_hallway, tag_target
        # Convert display → “world” coords (pre‐scale) for storage
        info = images_by_floor.get(current_floor)
        s = info["scale"] if info else init_scale
        dx = info["dx"] if info else 0
        dy = info["dy"] if info else 0

        # Translate back to “world” coords: (x_disp - dx) / s, (y_disp - dy) / s
        ox = (x_disp - dx) / s
        oy = (y_disp - dy) / s

        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == "normal":
                nid = f"N{len(data['nodes'])}_{int(ox)}_{int(oy)}"
                data["nodes"].append(
                    {
                        "id": nid,
                        "x": ox,
                        "y": oy,
                        "floor": current_floor,
                        "tags": {"hallway": False, "elevator": False, "full": False},
                        "wing": None,
                        "binType": "N",
                    }
                )
                tag_target = None

            elif mode == "hallway":
                nid = f"N{len(data['nodes'])}_{int(ox)}_{int(oy)}"
                data["nodes"].append(
                    {
                        "id": nid,
                        "x": ox,
                        "y": oy,
                        "floor": current_floor,
                        "tags": {"hallway": True, "elevator": False, "full": False},
                        "wing": None,
                        "binType": "N",
                    }
                )
                if last_hallway and last_hallway != nid:
                    data["edges"].append([last_hallway, nid, current_floor])
                last_hallway = nid
                tag_target = None

            elif mode == "branch":
                nid = f"N{len(data['nodes'])}_{int(ox)}_{int(oy)}"
                data["nodes"].append(
                    {
                        "id": nid,
                        "x": ox,
                        "y": oy,
                        "floor": current_floor,
                        "tags": {"hallway": False, "elevator": False, "full": False},
                        "wing": None,
                        "binType": "N",
                    }
                )
                h = nearest_hallway_node((x_disp, y_disp))
                if h:
                    data["edges"].append([nid, h["id"], current_floor])
                tag_target = None

            elif mode == "edge":
                n = nearest_node((x_disp, y_disp))
                if n and n["id"] not in edge_sel:
                    edge_sel.append(n["id"])
                    if len(edge_sel) == 2:
                        data["edges"].append([edge_sel[0], edge_sel[1], current_floor])
                        edge_sel.clear()
                tag_target = None

            elif mode == "tag":
                n = nearest_node((x_disp, y_disp))
                tag_target = n["id"] if n else None

            redraw()

        elif event == cv2.EVENT_RBUTTONDOWN and mode in ("normal", "hallway", "branch"):
            n = nearest_node((x_disp, y_disp))
            if n:
                # remove node & all edges connected on this floor
                data["nodes"].remove(n)
                data["edges"] = [
                    (u, v, fl)
                    for u, v, fl in data["edges"]
                    if not (
                        fl == current_floor and (u == n["id"] or v == n["id"])
                    )
                ]
                if mode == "hallway" and n["id"] == last_hallway:
                    last_hallway = None
                if mode == "tag" and n["id"] == tag_target:
                    tag_target = None
                redraw()

    # ----------------------------------
    # OpenCV window & callback binding
    # ----------------------------------
    cv2.namedWindow("editor", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback("editor", on_click)
    redraw()

    print(
        "Modes: n normal | h hallway | c continue | b branch | e edge | t tag | s save | q quit"
    )
    print(
        "In tag mode: click a node, then press 1=hallway,2=elevator,3=full, A–Z=wing, 1–4=binType (1=N,2=L,3=B,4=M)"
    )
    print("Switch floors: [ / ]  |  Zoom image: + / -")

    import numpy as np

    while True:
        cv2.imshow("editor", canvas)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("n"):
            mode = "normal"
            edge_sel.clear()
            last_hallway = None
            tag_target = None
        elif k == ord("h") or k == ord("c"):
            mode = "hallway"
            edge_sel.clear()
            tag_target = None
        elif k == ord("b"):
            mode = "branch"
            edge_sel.clear()
            tag_target = None
        elif k == ord("e"):
            mode = "edge"
            edge_sel.clear()
            tag_target = None
        elif k == ord("t"):
            mode = "tag"
            edge_sel.clear()
            tag_target = None

        # In tag mode, toggle tags or set wing/binType
        elif mode == "tag" and tag_target:
            n = next((x for x in data["nodes"] if x["id"] == tag_target), None)
            if n:
                # Toggle hallway/elevator/full with keys 1/2/3
                if k == ord("1"):
                    tags = n.setdefault("tags", {"hallway": False, "elevator": False, "full": False})
                    tags["hallway"] = not tags.get("hallway", False)
                elif k == ord("2"):
                    tags = n.setdefault("tags", {"hallway": False, "elevator": False, "full": False})
                    tags["elevator"] = not tags.get("elevator", False)
                elif k == ord("3"):
                    tags = n.setdefault("tags", {"hallway": False, "elevator": False, "full": False})
                    tags["full"] = not tags.get("full", False)
                # Wing with letters A–Z
                elif 65 <= k <= 90 or 97 <= k <= 122:
                    ch = chr(k).upper()
                    n["wing"] = ch
                # BinType with keys 1–4: 1=N, 2=L, 3=B, 4=M
                elif k == ord("1"):
                    n["binType"] = "N"
                elif k == ord("2"):
                    n["binType"] = "L"
                elif k == ord("3"):
                    n["binType"] = "B"
                elif k == ord("4"):
                    n["binType"] = "M"
                redraw()

        # Save JSON
        elif k == ord("s"):
            # Clean up edges so they’re lists, not tuples
            data["edges"] = [[u, v, fl] for (u, v, fl) in data["edges"]]
            json_path.write_text(json.dumps(data, indent=2))
            print(f"Saved {json_path}")

        # Switch floors
        elif k == ord("["):
            current_floor -= 1
            last_hallway = None
            tag_target = None
            if current_floor < 0:
                current_floor = 0
            redraw()
        elif k == ord("]"):
            current_floor += 1
            last_hallway = None
            tag_target = None
            redraw()

        # Zoom image of current floor
        elif k == ord("+"):
            info = images_by_floor.get(current_floor)
            if info:
                info["scale"] = info["scale"] * 1.1
                redraw()
        elif k == ord("-"):
            info = images_by_floor.get(current_floor)
            if info:
                info["scale"] = info["scale"] / 1.1
                redraw()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
