#!/usr/bin/env python3
"""
floorplan_editor.py — interactive multi‐floor floorplan node/edge editor

Standards:
  • “floor” lives only inside each node dict.
  • Each edge is exactly [u_id, v_id] (two elements).
  • We draw an edge only if both endpoints’ node["floor"] == current_floor.

Controls (all mode‐keys require Shift + <letter>):
  N: Normal (create node at click)
  H: Hallway (auto‐chain)
  C: Continue‐hallway (alias for H)
  B: Branch (auto‐link to nearest hallway)
  E: Edge (click two nodes to link)
  T: Tag mode:
       • Single‐click selects a node.
       • Click+drag box to select multiple nodes.
       • After selection, press:
           1 = toggle hallway tag
           2 = toggle elevator tag
           3 = toggle full tag
           A–Z = set wing
           1–4 = set binType (1=N,2=L,3=B,4=M)
  X: Delete‐edge mode (click near an edge to remove it)
  V: View full JSON in console
  [: switch to previous floor
  ]: switch to next floor
  +: increase current floor’s background scale
  -: decrease current floor’s background scale
  U: Upload image for current floor
  S: Save JSON
  Q: Quit
"""

import cv2
import json
import pathlib
import argparse
import numpy as np
import tkinter as tk
from tkinter import filedialog

# ---------------- Appearance constants ----------------
NODE_RADIUS  = 8       # base radius (in “world” units) before scaling
CLICK_THRESH = 15      # pix² threshold for “nearest node” detection
EDGE_COLOR   = (50, 150, 255)
EDGE_THICK   = 3

NODE_FILL    = (0, 255, 255)
NODE_OUTLINE = (0,   0,   0)
SEL_OUTLINE  = (0,   0, 255)

TAG_COLOR    = (0,   0, 255)   # for “H”, “E”, “F” badge
WING_COLOR   = (0, 255,   0)   # for wing label
BIN_COLOR    = (255, 0,   0)   # for binType label
# -----------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Interactive multi‐floor floorplan editor")
    p.add_argument(
        "--image",
        action="append",
        default=[],
        help="specify floor:image_path, e.g. 0:floor0.png; can repeat for each floor",
    )
    p.add_argument(
        "--json",
        required=True,
        help="path to nodes/edges JSON file (will be loaded/saved)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="initial UI zoom factor for background images",
    )
    return p.parse_args()

def main():
    args = parse_args()
    json_path = pathlib.Path(args.json)
    init_scale = args.scale

    # 1. Build images_by_floor from --image flags
    images_by_floor: dict[int, dict] = {}
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
                "disp": None,
                "dx": 0,
                "dy": 0,
            }
        except Exception as e:
            print(f"Invalid --image spec '{spec}': {e}")

    # 2. Load or initialize JSON data structure
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return
    else:
        data = {"nodes": [], "edges": []}

    data.setdefault("nodes", [])
    data.setdefault("edges", [])

    # 2a. Migrate any old “third‐element floor” inside edges by stripping it out:
    clean_edges = []
    for e in data["edges"]:
        if isinstance(e, list) and len(e) >= 2:
            clean_edges.append([e[0], e[1]])
    data["edges"] = clean_edges

    # 3. Track current floor and UI state
    current_floor = 0
    mode = "normal"            # normal/hallway/branch/edge/tag/delete-edge
    delete_edge_mode = False   # if True, click tries to delete the nearest edge
    edge_sel: list[str] = []   # for edge‐mode: store first clicked node ID
    last_hallway: str | None = None
    tag_target: list[str] = []     # can hold multiple IDs in box‐select
    box_selecting = False      # are we dragging a tag‐box?
    box_start = (0, 0)
    box_end = (0, 0)

    # ----------------------------------------------------
    # Helper: compute distance from point to segment
    # ----------------------------------------------------
    def dist_point_to_segment(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == dy == 0:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5

    # ----------------------------------------------------
    # Helper: scale & pan background, nearest node
    # ----------------------------------------------------
    def get_scaled_display_img(floor: int):
        info = images_by_floor.get(floor)
        if not info:
            return None
        orig = info["orig"]
        s = info["scale"]
        h, w = info["h"], info["w"]
        disp = cv2.resize(orig, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
        dx, dy = int(info["dx"]), int(info["dy"])
        canvas_img = cv2.copyMakeBorder(
            disp,
            top=max(0, dy),
            bottom=max(0, -dy),
            left=max(0, dx),
            right=max(0, -dx),
            borderType=cv2.BORDER_CONSTANT,
            value=(200, 200, 200),
        )
        x0 = max(0, -dx)
        y0 = max(0, -dy)
        canvas_img = canvas_img[y0 : y0 + int(h * s), x0 : x0 + int(w * s)]
        info["disp"] = canvas_img
        return canvas_img

    def nodes_on_floor():
        return [n for n in data["nodes"] if n.get("floor", 0) == current_floor]

    def edges_on_floor():
        """
        Return a list of (u, v) pairs for edges whose endpoints
        both have node['floor'] == current_floor.
        """
        node_by_id = {n["id"]: n for n in data["nodes"]}
        valid = []
        for e in data["edges"]:
            if not isinstance(e, list) or len(e) < 2:
                continue
            u, v = e[0], e[1]
            nu = node_by_id.get(u)
            nv = node_by_id.get(v)
            if nu is None or nv is None:
                continue
            if nu.get("floor", 0) == current_floor and nv.get("floor", 0) == current_floor:
                valid.append((u, v))
        return valid

    def id2pt_scaled():
        """
        Map each node ID → (x * scale + dx, y * scale + dy) in display coords.
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
        Return the nearest node on this floor within CLICK_THRESH of pt_disp.
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
        Among nodes on this floor with tags['hallway']==True,
        return the one whose display coords are closest to pt_disp.
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

    # -----------------
    # Redraw everything
    # -----------------
    canvas = None

    def redraw():
        nonlocal canvas
        disp_bg = get_scaled_display_img(current_floor)
        if disp_bg is None:
            # Blank gray if no background for this floor
            h, w = 600, 800
            canvas = 50 * np.ones((h, w, 3), dtype=np.uint8)
        else:
            canvas = disp_bg.copy()

        pts = id2pt_scaled()
        info = images_by_floor.get(current_floor)
        s = info["scale"] if info else init_scale
        # Dynamically scale the node radius
        disp_radius = max(2, int(NODE_RADIUS * s))

        # 1) Draw edges
        for u, v in edges_on_floor():
            if u in pts and v in pts:
                cv2.line(canvas, pts[u], pts[v], EDGE_COLOR, EDGE_THICK)

        # 2) Draw nodes
        for n in nodes_on_floor():
            x_disp, y_disp = pts[n["id"]]
            # fill circle
            cv2.circle(canvas, (x_disp, y_disp), disp_radius, NODE_FILL, -1)
            # outline
            cv2.circle(canvas, (x_disp, y_disp), disp_radius, NODE_OUTLINE, 1)

            # highlight in edge mode if selected
            if mode == "edge" and n["id"] in edge_sel:
                cv2.circle(canvas, (x_disp, y_disp), disp_radius + 4, SEL_OUTLINE, 2)
            # highlight in tag mode if in tag_target set
            if mode == "tag" and n["id"] in tag_target:
                cv2.circle(canvas, (x_disp, y_disp), disp_radius + 6, SEL_OUTLINE, 2)

            # draw tag badges (H/E/F)
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
                    (x_disp + disp_radius + 2, y_disp - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    TAG_COLOR,
                    1,
                )

            # draw wing label above node
            wing = n.get("wing")
            if wing:
                cv2.putText(
                    canvas,
                    wing.upper(),
                    (x_disp - 8, y_disp - disp_radius - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    WING_COLOR,
                    1,
                )

            # draw binType label below node
            binType = n.get("binType", "N")
            if binType:
                cv2.putText(
                    canvas,
                    binType,
                    (x_disp - 6, y_disp + disp_radius + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    BIN_COLOR,
                    1,
                )

        # 3) If box‐selecting, draw the marquee rectangle
        if box_selecting:
            x0, y0 = box_start
            x1, y1 = box_end
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (200, 200, 200), 1)

        # 4) Current floor text
        cv2.putText(
            canvas,
            f"Floor: {current_floor}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # ---------------------
    # Mouse callback (click + drag)
    # ---------------------
    def on_click(event, x_disp, y_disp, flags, param):
        nonlocal last_hallway, tag_target, delete_edge_mode, box_selecting, box_start, box_end

        # Map display coords → world coords
        info = images_by_floor.get(current_floor)
        s = info["scale"] if info else init_scale
        dx = info["dx"] if info else 0
        dy = info["dy"] if info else 0
        ox = (x_disp - dx) / s
        oy = (y_disp - dy) / s

        # 1) If in delete-edge mode, a left-click removes nearest edge
        if delete_edge_mode and event == cv2.EVENT_LBUTTONDOWN:
            pts = id2pt_scaled()
            best_edge = None
            best_dist = float("inf")
            for (u, v) in edges_on_floor():
                if u in pts and v in pts:
                    x1, y1 = pts[u]
                    x2, y2 = pts[v]
                    d = dist_point_to_segment(x_disp, y_disp, x1, y1, x2, y2)
                    if d < best_dist:
                        best_dist = d
                        best_edge = (u, v)
            if best_edge and best_dist <= 10:
                data["edges"] = [
                    e for e in data["edges"]
                    if not (e[0] == best_edge[0] and e[1] == best_edge[1])
                ]
                print(f"Deleted edge {best_edge[0]}–{best_edge[1]} on floor {current_floor}")
                redraw()
            delete_edge_mode = False
            return

        # 2) Tag‐mode box selection
        if mode == "tag":
            if event == cv2.EVENT_LBUTTONDOWN:
                # Start marquee
                box_start = (x_disp, y_disp)
                box_end = (x_disp, y_disp)
                box_selecting = True
                tag_target = []  # clear any prior selection
                return

            elif event == cv2.EVENT_MOUSEMOVE and box_selecting:
                box_end = (x_disp, y_disp)
                redraw()
                return

            elif event == cv2.EVENT_LBUTTONUP and box_selecting:
                x0, y0 = box_start
                x1, y1 = (x_disp, y_disp)
                xmin, xmax = sorted([x0, x1])
                ymin, ymax = sorted([y0, y1])
                pts = id2pt_scaled()
                selected = []
                for n in nodes_on_floor():
                    px, py = pts[n["id"]]
                    if xmin <= px <= xmax and ymin <= py <= ymax:
                        selected.append(n["id"])
                tag_target = selected[:]
                box_selecting = False
                redraw()
                return

        # 3) Regular click logic (node creation, edge mode, tag single-click, etc.)
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
                tag_target = []
                redraw()
                return

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
                    data["edges"].append([last_hallway, nid])
                last_hallway = nid
                tag_target = []
                redraw()
                return

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
                    data["edges"].append([nid, h["id"]])
                tag_target = []
                redraw()
                return

            elif mode == "edge":
                n = nearest_node((x_disp, y_disp))
                if n and n["id"] not in edge_sel:
                    edge_sel.append(n["id"])
                    if len(edge_sel) == 2:
                        data["edges"].append([edge_sel[0], edge_sel[1]])
                        edge_sel.clear()
                tag_target = []
                redraw()
                return

            elif mode == "tag":
                # Single‐click selection if user didn’t drag
                n = nearest_node((x_disp, y_disp))
                tag_target = [n["id"]] if n else []
                redraw()
                return

        # 4) Right‐click: delete a node (in modes normal/hallway/branch)
        elif event == cv2.EVENT_RBUTTONDOWN and mode in ("normal", "hallway", "branch"):
            n = nearest_node((x_disp, y_disp))
            if n:
                data["nodes"].remove(n)
                # Remove any edges that touch this node (any floor)
                data["edges"] = [e for e in data["edges"] if e[0] != n["id"] and e[1] != n["id"]]
                if mode == "hallway" and n["id"] == last_hallway:
                    last_hallway = None
                if n["id"] in tag_target:
                    tag_target.remove(n["id"])
                redraw()
            return

    # ----------------------
    # Set up OpenCV window
    # ----------------------
    cv2.namedWindow("editor", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback("editor", on_click)
    redraw()

    print(
        "Modes (Shift+key):\n"
        "  N=Normal | H=Hallway | C=Continue‐Hallway | B=Branch | E=Edge | T=Tag\n"
        "  X=Delete‐Edge | V=View‐JSON | S=Save | Q=Quit\n"
        "Switch floors: [ / ]   |   Zoom: + / -   |   Upload image: U\n"
        "Tag box: click+drag to select multiple nodes → then press\n"
        "  1=toggle hallway, 2=toggle elevator, 3=toggle full,\n"
        "  A–Z=set wing, 1–4=set binType (1=N,2=L,3=B,4=M)"
    )

    while True:
        cv2.imshow("editor", canvas)
        k = cv2.waitKey(20) & 0xFF

        if k == ord("q"):  # Quit
            break

        # ---------------- Mode switches (uppercase only) ----------------
        elif k == ord("n"):
            mode = "normal"
            delete_edge_mode = False
            edge_sel.clear()
            last_hallway = None
            tag_target = []

        elif k == ord("h") or k == ord("C"):
            mode = "hallway"
            delete_edge_mode = False
            edge_sel.clear()
            tag_target = []

        elif k == ord("l"):
            mode = "branch"
            delete_edge_mode = False
            edge_sel.clear()
            tag_target = []

        elif k == ord("u"):
            mode = "edge"
            delete_edge_mode = False
            edge_sel.clear()
            tag_target = []

        elif k == ord("t"):
            mode = "tag"
            delete_edge_mode = False
            edge_sel.clear()
            tag_target = []
            print("Tag mode: click or click+drag to select nodes for tagging.")

        elif k == ord("x"):
            delete_edge_mode = True
            tag_target = []
            edge_sel.clear()
            print("Delete‐edge mode: click near an edge to remove it.")

        elif k == ord("v"):
            # Print full JSON to console
            print(json.dumps(data, indent=2))

        # ---------------- Save JSON ----------------
        elif k == ord("s"):
            # We keep nodes["floor"], and edges remain two‐element lists
            json_path.write_text(json.dumps(data, indent=2))
            print(f"Saved {json_path}")

        # ---------------- Upload image (temporary Tk) ----------------
        elif k == ord("i"):
            # create a short‐lived Tk root so right‐click in CV window does nothing
            dlg = tk.Tk()
            dlg.withdraw()
            path = filedialog.askopenfilename(
                title=f"Select image for floor {current_floor}",
                filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")],
            )
            dlg.destroy()
            if path:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Error: could not load image {path}")
                else:
                    h, w = img.shape[:2]
                    images_by_floor[current_floor] = {
                        "orig": img,
                        "scale": init_scale,
                        "w": w,
                        "h": h,
                        "disp": None,
                        "dx": 0,
                        "dy": 0,
                    }
                    print(f"Loaded image for floor {current_floor}: {path}")
                    redraw()

        # ---------------- Switch floors ----------------
        elif k == ord("["):
            current_floor = max(0, current_floor - 1)
            delete_edge_mode = False
            last_hallway = None
            tag_target = []
            redraw()

        elif k == ord("]"):
            current_floor += 1
            delete_edge_mode = False
            last_hallway = None
            tag_target = []
            redraw()

        # ---------------- Zoom background (and scale nodes accordingly) ----------------
        elif k == ord("+"):
            info = images_by_floor.get(current_floor)
            if info:
                info["scale"] *= 1.1
                redraw()

        elif k == ord("-"):
            info = images_by_floor.get(current_floor)
            if info:
                info["scale"] /= 1.1
                redraw()

        # ---------------- Tag‐mode keystrokes ----------------
        elif mode == "tag" and tag_target:
            # Toggle hallway/elevator/full
            if k == ord("1"):
                for nid in tag_target:
                    n = next((x for x in data["nodes"] if x["id"] == nid), None)
                    if n:
                        tags = n.setdefault("tags", {"hallway": False, "elevator": False, "full": False})
                        tags["hallway"] = not tags.get("hallway", False)
                redraw()

            elif k == ord("2"):
                for nid in tag_target:
                    n = next((x for x in data["nodes"] if x["id"] == nid), None)
                    if n:
                        tags = n.setdefault("tags", {"hallway": False, "elevator": False, "full": False})
                        tags["elevator"] = not tags.get("elevator", False)
                redraw()

            elif k == ord("3"):
                for nid in tag_target:
                    n = next((x for x in data["nodes"] if x["id"] == nid), None)
                    if n:
                        tags = n.setdefault("tags", {"hallway": False, "elevator": False, "full": False})
                        tags["full"] = not tags.get("full", False)
                redraw()

            # Wing with A–Z
            elif 65 <= k <= 90 or 97 <= k <= 122:
                ch = chr(k).upper()
                for nid in tag_target:
                    n = next((x for x in data["nodes"] if x["id"] == nid), None)
                    if n:
                        n["wing"] = ch
                redraw()

            # BinType with numeric keys 1–4 (NB: 1 also toggles hallway; choose 2/3/4 here)
            elif k == ord("1"):
                # If user truly wants binType “N”, they can press “1” again.
                for nid in tag_target:
                    n = next((x for x in data["nodes"] if x["id"] == nid), None)
                    if n:
                        n["binType"] = "N"
                redraw()
            elif k == ord("2"):
                for nid in tag_target:
                    n = next((x for x in data["nodes"] if x["id"] == nid), None)
                    if n:
                        n["binType"] = "L"
                redraw()
            elif k == ord("3"):
                for nid in tag_target:
                    n = next((x for x in data["nodes"] if x["id"] == nid), None)
                    if n:
                        n["binType"] = "B"
                redraw()
            elif k == ord("4"):
                for nid in tag_target:
                    n = next((x for x in data["nodes"] if x["id"] == nid), None)
                    if n:
                        n["binType"] = "M"
                redraw()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
