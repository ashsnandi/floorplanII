#!/usr/bin/env python3
"""
floorplan_editor.py — interactive multi‐floor floorplan editor with independent
background‐image overlay and world‐zoom/pan.

Features:
  • Nodes and edges live in “world” coordinates.
  • “ViewPan” mode drags the camera over the entire world (nodes, edges, background).
  • “ImagePan” mode drags only the background image (nodes remain fixed in world).
  • Clicking in node/edge/tag modes always places/selects using world coords =
    (view_x + screen_x/ view_scale, view_y + screen_y/ view_scale).
  • Zooming (',' to zoom out, '.' to zoom in) scales the view without affecting
    node positions or background size.
  • Background‐image scaling ('+' / '-') scales the image itself (in world units),
    under the same origin.
  • Save/load JSON, switch floors, tag nodes, delete edges, etc.

Controls (uppercase for mode‐switch; lowercase in tag mode):
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
  Z: View full JSON in console
  [: switch to previous floor
  ]: switch to next floor
  +: increase current floor’s background‐image scale
  -: decrease current floor’s background‐image scale
  U: Upload image for current floor
  S: Save JSON
  Q: Quit

Pan/zoom modes:
  V: Enter ImagePan (drag moves background image only)
  M: Enter ViewPan  (drag moves camera over world: nodes+image)
  ,: Zoom view out (view_scale /= 1.1)
  .: Zoom view in  (view_scale *= 1.1)

Window size is fixed at WIN_W×WIN_H (default 1920×1080).
"""

import cv2
import json
import pathlib
import argparse
import numpy as np
import tkinter as tk
from tkinter import filedialog

# ---------------- Appearance constants ----------------
NODE_RADIUS       = 8       # on‐screen radius in pixels (before view zoom)
CLICK_THRESH      = 15      # pixel‐distance threshold for “nearest node”
EDGE_COLOR        = (50, 150, 255)
EDGE_THICK        = 3

NODE_FILL         = (0, 255, 255)
NODE_OUTLINE      = (0,   0,   0)
SEL_OUTLINE       = (0,   0, 255)

TAG_COLOR         = (0,   0, 255)   # for “H”, “E”, “F” badge
WING_COLOR        = (0, 255,   0)   # for wing label
BIN_COLOR         = (255, 0,   0)   # for binType label

WIN_W, WIN_H      = 1920, 1080
# -----------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Interactive multi‐floor floorplan editor")
    p.add_argument(
        "--image",
        action="append",
        default=[],
        help="specify floor:image_path, e.g. 0:floor0.png; can repeat per floor",
    )
    p.add_argument(
        "--json",
        required=True,
        help="path to nodes/edges JSON file (load/save)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="initial image‐scale factor (world units)",
    )
    return p.parse_args()

def main():
    offset_x = 0.0
    offset_y = 0.0
    args = parse_args()
    json_path = pathlib.Path(args.json)
    init_scale = args.scale

    # Initialize Tk for file dialogs
    tk_root = tk.Tk()
    tk_root.withdraw()

    # 1. Build images_by_floor from --image flags
    images_by_floor: dict[int, dict] = {}
    for spec in args.image:
        try:
            floor_str, img_path = spec.split(":", 1)
            fl = int(floor_str)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: failed to load {img_path} for floor {fl}")
                continue
            h, w = img.shape[:2]
            images_by_floor[fl] = {
                "orig": img,         # original OpenCV image
                "scale": init_scale, # background‐scale in world coords
                "w": w, "h": h,      # original image dimensions
                "disp": None,        # placeholder for scaled image
                "dx": 0.0,           # image offset X in world coords
                "dy": 0.0,           # image offset Y in world coords
            }
        except Exception as e:
            print(f"Ignoring bad --image spec '{spec}': {e}")

    # 2. Load or initialize JSON data
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

    # 2a. Strip any old “third element” from edges
    clean_edges = []
    for e in data["edges"]:
        if isinstance(e, list) and len(e) >= 2:
            clean_edges.append([e[0], e[1]])
    data["edges"] = clean_edges

    # 3. Track current floor and UI state
    current_floor = 0
    mode = "normal"            # normal/hallway/branch/edge/tag/delete-edge/ImagePan/ViewPan
    delete_edge_mode = False   # True for Shift+X delete‐edge
    edge_sel: list[str] = []   # for Edge mode
    last_hallway: str | None = None
    tag_target: list[str] = [] # selected node IDs in Tag mode (can be multiple)
    box_selecting = False      # for marquee in Tag mode
    box_start = (0, 0)
    box_end = (0, 0)

    #  Background lock (prevents ImagePan)
    bg_locked = False

    #  ImagePan state
    img_panning = False
    img_pan_start = (0, 0)
    orig_img_dx = 0.0
    orig_img_dy = 0.0

    #  View (camera) pan/zoom
    view_scale = 1.0
    view_x = 0.0
    view_y = 0.0

    #  ViewPan state
    view_panning = False
    view_pan_start = (0, 0)
    orig_view_x = 0.0
    orig_view_y = 0.0

    # ----------------------------------------------------
    # Helper: distance from point to segment
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
    # Helper: get scaled background image for floor
    # ----------------------------------------------------
    def get_scaled_display_img(floor: int):
        info = images_by_floor.get(floor)
        if not info:
            return None
        orig = info["orig"]
        s = info["scale"]
        h, w = info["h"], info["w"]
        disp = cv2.resize(orig, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
        info["disp"] = disp
        return disp

    def nodes_on_floor():
        return [n for n in data["nodes"] if n.get("floor", 0) == current_floor]

    def edges_on_floor():
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

    def id2pt_world():
        """
        Map node ID → (world_x, world_y).
        """
        pts = {}
        for n in nodes_on_floor():
            pts[n["id"]] = (n["x"], n["y"])
        return pts

    def id2pt_view():
        """
        Map node ID → (screen_x, screen_y) after applying:
          1) the background image’s own (dx,dy) shift
          2) the camera’s (view_x,view_y) pan
          3) the camera’s view_scale zoom

        Formula:
          screen_x = ((world_x + img_dx + offset_x) - view_x) * view_scale
          screen_y = ((world_y + img_dy + offset_y) - view_y) * view_scale
        """
        pts_v = {}
        info = images_by_floor.get(current_floor)
        img_dx = info["dx"] if info else 0.0
        img_dy = info["dy"] if info else 0.0

        for n in nodes_on_floor():
            wx, wy = n["x"], n["y"]
            sx = int(((wx + img_dx + offset_x) - view_x) * view_scale)
            sy = int(((wy + img_dy + offset_y) - view_y) * view_scale)
            pts_v[n["id"]] = (sx, sy)
        return pts_v

    def nearest_node_view(click_x, click_y):
        """
        Among nodes on this floor, find the one whose screen‐distance to (click_x,click_y)
        is within CLICK_THRESH. Return that node or None.
        """
        pts_v = id2pt_view()
        for n in nodes_on_floor():
            sx, sy = pts_v[n["id"]]
            dx = sx - click_x
            dy = sy - click_y
            if dx * dx + dy * dy <= CLICK_THRESH * CLICK_THRESH:
                return n
        return None

    def nearest_hallway_node_view(click_x, click_y):
        pts_v = id2pt_view()
        best, bd = None, float("inf")
        for n in nodes_on_floor():
            if n.get("tags", {}).get("hallway"):
                sx, sy = pts_v[n["id"]]
                d = (sx - click_x) ** 2 + (sy - click_y) ** 2
                if d < bd:
                    best, bd = n, d
        return best

    # -----------------
    # Redraw everything
    # -----------------
    canvas = None

    def redraw():
        nonlocal canvas, offset_x, offset_y

        # 1) Fetch (and cache) the scaled background image for the current floor.
        #    If there is no image for this floor, get_scaled_display_img() returns None.
        disp_bg = get_scaled_display_img(current_floor)

        if disp_bg is None:
            # No background loaded → create a default gray “world” of fixed size.
            world_h, world_w = 2000, 2000
            world = 50 * np.ones((world_h, world_w, 3), dtype=np.uint8)

            # In this branch, offset_x/offset_y don’t matter because there’s no image to place.
            def with_offset(wx, wy):
                return wx + dx_w + offset_x, wy + dy_w + offset_y

        else:
            # We do have a background.  disp_bg is the resized image (a NumPy array).
            img = disp_bg
            img_h, img_w = img.shape[:2]
            info = images_by_floor[current_floor]

            # Use the float dx/dy from the image-storage dictionary
            dx_w = info["dx"]
            dy_w = info["dy"]

            # Compute the bounding box (in floating “world” units) that encloses:
            #   • the background image (which runs from 0→img_w, 0→img_h, but shifted by dx_w,dy_w)
            #   • all nodes (their (x,y) lives in the same “world” coordinate system)
            min_x = min(0.0, dx_w)
            min_y = min(0.0, dy_w)
            max_x = max(img_w + dx_w, 0.0)
            max_y = max(img_h + dy_w, 0.0)

            for n in nodes_on_floor():
                wx, wy = float(n["x"]), float(n["y"])
                if wx < min_x: min_x = wx
                if wy < min_y: min_y = wy
                if wx > max_x: max_x = wx
                if wy > max_y: max_y = wy

            # Add a constant margin so that nothing ever falls outside the canvas.
            margin = 200.0
            world_w_f = (max_x - min_x) + margin   # float width
            world_h_f = (max_y - min_y) + margin   # float height

            # Convert to integers for building a NumPy array
            world_w = int(max(1, world_w_f))
            world_h = int(max(1, world_h_f))

            # Recompute offset_x/offset_y in floats—this ensures that the top‐left of our “world” is at (min_x, min_y).
            offset_x = -min_x + (margin / 2.0)
            offset_y = -min_y + (margin / 2.0)

            # Create a gray “world” canvas of size (world_h × world_w)
            world = 50 * np.ones((world_h, world_w, 3), dtype=np.uint8)

            # Draw the background image onto that canvas at pixel coordinates:
            #   (dx_w + offset_x, dy_w + offset_y)
            bx = dx_w + offset_x
            by = dy_w + offset_y
            x0, y0 = int(bx), int(by)
            x1, y1 = int(bx + img_w), int(by + img_h)

            if 0 <= x0 < world_w and 0 <= y0 < world_h:
                ox0 = max(x0, 0)
                oy0 = max(y0, 0)
                ox1 = min(x1, world_w)
                oy1 = min(y1, world_h)
                sx0 = ox0 - x0
                sy0 = oy0 - y0
                sx1 = sx0 + (ox1 - ox0)
                sy1 = sy0 + (oy1 - oy0)
                world[oy0:oy1, ox0:ox1] = img[sy0:sy1, sx0:sx0 + (ox1 - ox0)]

            # After drawing the background, define a helper function so we can draw nodes/edges
            # by shifting their world‐coordinates by (offset_x, offset_y).
            def with_offset(wx, wy):
                return wx + offset_x, wy + offset_y

        # 2) Draw edges on the “world” canvas.  All coordinates must be ints!
        if disp_bg is not None:
            for (u, v) in edges_on_floor():
                nu = next(n for n in data["nodes"] if n["id"] == u)
                nv = next(n for n in data["nodes"] if n["id"] == v)

                # Compute float positions, then immediately cast to int
                fx1, fy1 = with_offset(int(nu["x"]), int(nu["y"]))
                fx2, fy2 = with_offset(int(nv["x"]), int(nv["y"]))
                x1o, y1o = int(fx1), int(fy1)
                x2o, y2o = int(fx2), int(fy2)

                cv2.line(world, (x1o, y1o), (x2o, y2o), EDGE_COLOR, EDGE_THICK)
        else:
            for (u, v) in edges_on_floor():
                nu = next(n for n in data["nodes"] if n["id"] == u)
                nv = next(n for n in data["nodes"] if n["id"] == v)
                x1o, y1o = int(nu["x"]), int(nu["y"])
                x2o, y2o = int(nv["x"]), int(nv["y"])
                cv2.line(world, (x1o, y1o), (x2o, y2o), EDGE_COLOR, EDGE_THICK)

        # 3) Draw nodes on the “world” canvas
        if disp_bg is not None:
            for n in nodes_on_floor():
                fx, fy = with_offset(int(n["x"]), int(n["y"]))
                wo_x, wo_y = int(fx), int(fy)

                cv2.circle(world, (wo_x, wo_y), NODE_RADIUS, NODE_FILL, -1)
                cv2.circle(world, (wo_x, wo_y), NODE_RADIUS, NODE_OUTLINE, 1)

                if mode == "edge" and n["id"] in edge_sel:
                    cv2.circle(world, (wo_x, wo_y), NODE_RADIUS + 4, SEL_OUTLINE, 2)
                if mode == "tag" and n["id"] in tag_target:
                    cv2.circle(world, (wo_x, wo_y), NODE_RADIUS + 6, SEL_OUTLINE, 2)

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
                        world,
                        badge,
                        (wo_x + NODE_RADIUS + 2, wo_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        TAG_COLOR,
                        1,
                    )

                wing = n.get("wing")
                if wing:
                    cv2.putText(
                        world,
                        wing.upper(),
                        (wo_x - 8, wo_y - NODE_RADIUS - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        WING_COLOR,
                        1,
                    )

                binType = n.get("binType", "N")
                if binType:
                    cv2.putText(
                        world,
                        binType,
                        (wo_x - 6, wo_y + NODE_RADIUS + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        BIN_COLOR,
                        1,
                    )
        else:
            for n in nodes_on_floor():
                x1o, y1o = int(n["x"]), int(n["y"])
                cv2.circle(world, (x1o, y1o), NODE_RADIUS, NODE_FILL, -1)
                cv2.circle(world, (x1o, y1o), NODE_RADIUS, NODE_OUTLINE, 1)

                if mode == "edge" and n["id"] in edge_sel:
                    cv2.circle(world, (x1o, y1o), NODE_RADIUS + 4, SEL_OUTLINE, 2)
                if mode == "tag" and n["id"] in tag_target:
                    cv2.circle(world, (x1o, y1o), NODE_RADIUS + 6, SEL_OUTLINE, 2)

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
                        world,
                        badge,
                        (x1o + NODE_RADIUS + 2, y1o - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        TAG_COLOR,
                        1,
                    )

                wing = n.get("wing")
                if wing:
                    cv2.putText(
                        world,
                        wing.upper(),
                        (x1o - 8, y1o - NODE_RADIUS - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        WING_COLOR,
                        1,
                    )

                binType = n.get("binType", "N")
                if binType:
                    cv2.putText(
                        world,
                        binType,
                        (x1o - 6, y1o + NODE_RADIUS + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        BIN_COLOR,
                        1,
                    )

        # 4) Crop the “world” canvas according to view_x/view_y and view_scale
        crop_w = int(WIN_W / view_scale)
        crop_h = int(WIN_H / view_scale)
        w_h, w_w = world.shape[:2]

        # Clamp view_x, view_y so the cropping rectangle stays inside “world.”
        vx = int(np.clip(view_x, 0, max(0, w_w - crop_w)))
        vy = int(np.clip(view_y, 0, max(0, w_h - crop_h)))

        cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        ox0, oy0 = max(vx, 0), max(vy, 0)
        ox1, oy1 = min(vx + crop_w, w_w), min(vy + crop_h, w_h)

        if ox0 < ox1 and oy0 < oy1:
            sub = world[oy0:oy1, ox0:ox1]
            sx0 = ox0 - vx
            sy0 = oy0 - vy
            cropped[sy0 : sy0 + (oy1 - oy0), sx0 : sx0 + (ox1 - ox0)] = sub

        # 5) Resize that cropped region up to exactly (WIN_W × WIN_H)
        view_canvas = cv2.resize(cropped, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)

        # 6) Draw status text
        status = f"Floor: {current_floor}  Mode: {mode}  BG‐Locked: {bg_locked}  View‐Zoom: {view_scale:.2f}"
        cv2.putText(
            view_canvas,
            status,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # 7) If box-selecting in tag mode, draw that selection rectangle
        if mode == "tag" and box_selecting:
            x0, y0 = box_start
            x1, y1 = box_end
            cv2.rectangle(view_canvas, (x0, y0), (x1, y1), (200, 200, 200), 1)

        # Finally, store the completed view in the global “canvas”
        canvas = view_canvas

    # ---------------------
    # Mouse callback
    # ---------------------
    def on_click(event, x, y, flags, param):
        nonlocal view_x, view_y, view_scale
        nonlocal last_hallway, tag_target, delete_edge_mode
        nonlocal box_selecting, box_start, box_end
        nonlocal img_panning, img_pan_start, orig_img_dx, orig_img_dy
        nonlocal view_panning, view_pan_start, orig_view_x, orig_view_y

        # 0) Convert (screen_x, screen_y) → (raw world_x, world_y),
        #    then subtract the background’s own dx/dy so that
        #    node.x/node.y end up in “image‐local” world coords.
        info = images_by_floor.get(current_floor)
        if info:
            img_dx = info["dx"]
            img_dy = info["dy"]
        else:
            img_dx = 0.0
            img_dy = 0.0

        raw_wx = view_x + (x / view_scale)
        raw_wy = view_y + (y / view_scale)

        world_x = raw_wx - img_dx - offset_x
        world_y = raw_wy - img_dy - offset_y

        # 1) IMAGE‐PAN mode (Shift+V)
        if mode == "ImagePan":
            if bg_locked:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                img_panning   = True
                img_pan_start = (x, y)
                orig_img_dx   = images_by_floor.get(current_floor, {}).get("dx", 0.0)
                orig_img_dy   = images_by_floor.get(current_floor, {}).get("dy", 0.0)
                return

            elif event == cv2.EVENT_MOUSEMOVE and img_panning:
                px, py = img_pan_start
                dx_screen = x - px
                dy_screen = y - py
                # Convert screen‐delta → world offset for the background image
                images_by_floor[current_floor]["dx"] = orig_img_dx + (dx_screen / view_scale)
                images_by_floor[current_floor]["dy"] = orig_img_dy + (dy_screen / view_scale)
                redraw()
                return

            elif event == cv2.EVENT_LBUTTONUP and img_panning:
                img_panning = False
                return
            else:
                return  # consume all clicks in ImagePan mode

        # 2) VIEW‐PAN mode (Shift+M)
        if mode == "ViewPan":
            if event == cv2.EVENT_LBUTTONDOWN:
                view_panning   = True
                view_pan_start = (x, y)
                orig_view_x    = view_x
                orig_view_y    = view_y
                return

            elif event == cv2.EVENT_MOUSEMOVE and view_panning:
                px, py = view_pan_start
                dx_screen = x - px
                dy_screen = y - py
                # Invert the screen‐delta into camera movement
                view_x = orig_view_x - (dx_screen / view_scale)
                view_y = orig_view_y - (dy_screen / view_scale)
                redraw()
                return

            elif event == cv2.EVENT_LBUTTONUP and view_panning:
                view_panning = False
                return
            else:
                return  # consume all clicks in ViewPan mode

        # 3) DELETE‐EDGE mode (Shift+X)
        if delete_edge_mode and event == cv2.EVENT_LBUTTONDOWN:
            pts_v      = id2pt_view()
            best_edge  = None
            best_dist  = float("inf")
            for (u, v) in edges_on_floor():
                if u in pts_v and v in pts_v:
                    x1, y1 = pts_v[u]
                    x2, y2 = pts_v[v]
                    d = dist_point_to_segment(x, y, x1, y1, x2, y2)
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

        # 4) TAG‐mode: single‐click vs box‐drag
        if mode == "tag":
            # a) LBUTTONDOWN: start potential box‐select
            if event == cv2.EVENT_LBUTTONDOWN:
                box_start     = (x, y)
                box_end       = (x, y)
                box_selecting = True
                return

            # b) MOUSEMOVE while dragging: update marquee
            elif event == cv2.EVENT_MOUSEMOVE and box_selecting:
                box_end = (x, y)
                redraw()
                return

            # c) LBUTTONUP: decide between “click” or “box‐select”
            elif event == cv2.EVENT_LBUTTONUP and box_selecting:
                x0, y0 = box_start
                x1, y1 = (x, y)
                dx_abs = abs(x1 - x0)
                dy_abs = abs(y1 - y0)

                # threshold for “zero‐drag” (treat as single‐click)
                if dx_abs <= CLICK_THRESH // 2 and dy_abs <= CLICK_THRESH // 2:
                    # Single‐click: find the nearest node under the cursor
                    n = nearest_node_view(x, y)
                    tag_target = [n["id"]] if (n is not None) else []
                    box_selecting = False
                    redraw()
                    return

                # Otherwise, it was a genuine drag → do box‐select
                xmin, xmax = sorted([x0, x1])
                ymin, ymax = sorted([y0, y1])
                pts_v      = id2pt_view()
                selected   = []
                for n in nodes_on_floor():
                    sx, sy = pts_v[n["id"]]
                    if xmin <= sx <= xmax and ymin <= sy <= ymax:
                        selected.append(n["id"])
                tag_target    = selected[:]
                box_selecting = False
                redraw()
                return

        # 5) Regular left‐click modes (N, H, B, E)
        if event == cv2.EVENT_LBUTTONDOWN:
            # N: Normal (create node at world_x,world_y)
            if mode == "normal":
                nid = f"N{len(data['nodes'])}_{int(world_x)}_{int(world_y)}"
                data["nodes"].append({
                    "id":    nid,
                    "x":     world_x,
                    "y":     world_y,
                    "floor": current_floor,
                    "tags":  {"hallway": False, "elevator": False, "full": False},
                    "wing":  None,
                    "binType": "N",
                })
                tag_target = []
                redraw()
                return

            # H or C: Hallway (auto‐chain)
            elif mode == "hallway":
                nid = f"N{len(data['nodes'])}_{int(world_x)}_{int(world_y)}"
                data["nodes"].append({
                    "id":    nid,
                    "x":     world_x,
                    "y":     world_y,
                    "floor": current_floor,
                    "tags":  {"hallway": True, "elevator": False, "full": False},
                    "wing":  None,
                    "binType": "N",
                })
                if last_hallway and last_hallway != nid:
                    data["edges"].append([last_hallway, nid])
                last_hallway = nid
                tag_target    = []
                redraw()
                return

            # B: Branch (auto‐link to nearest hallway)
            elif mode == "branch":
                nid = f"N{len(data['nodes'])}_{int(world_x)}_{int(world_y)}"
                data["nodes"].append({
                    "id":    nid,
                    "x":     world_x,
                    "y":     world_y,
                    "floor": current_floor,
                    "tags":  {"hallway": False, "elevator": False, "full": False},
                    "wing":  None,
                    "binType": "N",
                })
                h = nearest_hallway_node_view(x, y)
                if h:
                    data["edges"].append([nid, h["id"]])
                tag_target = []
                redraw()
                return

            # E: Edge (click two nodes to connect)
            elif mode == "edge":
                n = nearest_node_view(x, y)
                if n and n["id"] not in edge_sel:
                    edge_sel.append(n["id"])
                    if len(edge_sel) == 2:
                        data["edges"].append([edge_sel[0], edge_sel[1]])
                        edge_sel.clear()
                tag_target = []
                redraw()
                return

        # 6) Right‐click: Delete a node (in modes normal/hallway/branch)
        elif event == cv2.EVENT_RBUTTONDOWN and mode in ("normal", "hallway", "branch"):
            n = nearest_node_view(x, y)
            if n:
                data["nodes"].remove(n)
                # Remove any edges involving that node on any floor
                data["edges"] = [
                    e for e in data["edges"]
                    if not (e[0] == n["id"] or e[1] == n["id"])
                ]
                if mode == "hallway" and n["id"] == last_hallway:
                    last_hallway = None
                if n["id"] in tag_target:
                    tag_target.remove(n["id"])
                redraw()
            return

    # ----------------------
    # Window & callback setup
    # ----------------------
    cv2.namedWindow("editor", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("editor", WIN_W, WIN_H)
    cv2.setMouseCallback("editor", on_click)
    redraw()

    print(
        "Modes (Shift+key):\n"
        "  N=Normal | H=Hallway | C=Continue‐Hallway | B=Branch | E=Edge | T=Tag\n"
        "  X=Delete‐Edge | Z=View‐JSON | S=Save | Q=Quit\n"
        "  V=ImagePan (drag bg only) | M=ViewPan (drag world) | L=Toggle BG lock\n"
        "  F=Fit background to window | [, ] = switch floors | U=Upload image\n"
        "  BG‐zoom: + / -  |  View‐zoom: ',' / '.'\n"
        "Tag box: click+drag to select multiple nodes, then press\n"
        "  1=toggle hallway, 2=toggle elevator, 3=toggle full,\n"
        "  A–Z=set wing, 1–4=set binType (1=N,2=L,3=B,4=M)"
    )

    while True:
        cv2.imshow("editor", canvas)
        k = cv2.waitKey(20) & 0xFF

        if k == ord("Q"):  # Quit
            break

        # --- Mode switches ---
        elif k == ord("N"):
            mode = "normal"
            delete_edge_mode = False
            edge_sel.clear()
            last_hallway = None
            tag_target = []

        elif k in (ord("H"), ord("C")):
            mode = "hallway"
            delete_edge_mode = False
            edge_sel.clear()
            tag_target = []

        elif k == ord("B"):
            mode = "branch"
            delete_edge_mode = False
            edge_sel.clear()
            tag_target = []

        elif k == ord("E"):
            mode = "edge"
            delete_edge_mode = False
            edge_sel.clear()
            tag_target = []

        elif k == ord("T"):
            mode = "tag"
            delete_edge_mode = False
            edge_sel.clear()
            tag_target = []
            print("Tag mode: click or click+drag to select nodes for tagging.")

        elif k == ord("X"):
            delete_edge_mode = True
            tag_target = []
            edge_sel.clear()
            print("Delete‐edge mode: click near an edge to remove it.")

        elif k == ord("Z"):
            print(json.dumps(data, indent=2))

        elif k == ord("S"):
            json_path.write_text(json.dumps(data, indent=2))
            print(f"Saved {json_path}")

        elif k == ord("U"):
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
                        "dx": 0.0,
                        "dy": 0.0,
                    }
                    print(f"Loaded image for floor {current_floor}: {path}")
                    redraw()

        elif k == ord("V"):
            mode = "ImagePan"
            delete_edge_mode = False
            edge_sel.clear()
            tag_target = []
            print("ImagePan mode: drag background image around (toggle L to lock).")

        elif k == ord("M"):
            mode = "ViewPan"
            delete_edge_mode = False
            edge_sel.clear()
            tag_target = []
            print("ViewPan mode: drag camera over world (nodes+image).")

        elif k == ord("L"):
            bg_locked = not bg_locked
            print(f"Background locked = {bg_locked}")

        # Fit background to window (Shift+F)
        elif k == ord("F"):
            info = images_by_floor.get(current_floor)
            if info:
                img_w, img_h = info["w"], info["h"]
                scale_w = WIN_W / img_w
                scale_h = WIN_H / img_h
                new_scale = min(scale_w, scale_h)
                info["scale"] = new_scale
                info["dx"] = 0.0
                info["dy"] = 0.0
                view_x = 0.0
                view_y = 0.0
                print(f"Fitted floor {current_floor} to window (bg scale={new_scale:.3f})")
                redraw()

        # Switch floors
        elif k == ord("["):
            current_floor = max(0, current_floor - 1)
            delete_edge_mode = False
            last_hallway = None
            tag_target = []
            view_x = 0.0
            view_y = 0.0
            redraw()

        elif k == ord("]"):
            current_floor += 1
            delete_edge_mode = False
            last_hallway = None
            tag_target = []
            view_x = 0.0
            view_y = 0.0
            redraw()

        # Background zoom (Shift+'+' / Shift+'-')
        elif k == ord("+"):
            info = images_by_floor.get(current_floor)
            if info:
                info["scale"] *= 1.02
                redraw()
        elif k == ord("-"):
            info = images_by_floor.get(current_floor)
            if info:
                info["scale"] /= 1.02
                redraw()

        # View zoom out/in
        elif k == ord(","):
            view_scale /= 1.1
            redraw()
        elif k == ord("."):
            view_scale *= 1.1
            redraw()

        # Tag‐mode keystrokes
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

            # BinType with 1–4
            elif k == ord("9"):
                for nid in tag_target:
                    n = next((x for x in data["nodes"] if x["id"] == nid), None)
                    if n:
                        n["binType"] = "N"
                redraw()
            elif k == ord("8"):
                for nid in tag_target:
                    n = next((x for x in data["nodes"] if x["id"] == nid), None)
                    if n:
                        n["binType"] = "L"
                redraw()
            elif k == ord("7"):
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
