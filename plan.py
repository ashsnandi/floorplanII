#!/usr/bin/env python3
"""
floorplan_json_editor.py — interactive JSON-based floorplan editor

Features:
  • Load nodes/edges from JSON
  • Optionally overlay a background image (--image)
  • Auto-scale canvas based on JSON extents
  • Switch between floors ([ and ])
  • Select nodes and toggle tags: e elevator, h hallway, f full
  • Edge mode: add edges by selecting two nodes (key 'a')
  • Visual highlighting and save back to JSON
Controls:
  '[' / ']'  switch floor
  left-click  select node (or edge endpoint in edge mode)
  e           toggle elevator tag
  h           toggle hallway tag
  f           toggle full tag
  a           toggle edge mode (then click two nodes to connect)
  s           save JSON
  q           quit
"""
import cv2, json, pathlib, argparse
import numpy as np

# Appearance constants
NODE_RADIUS  = 8
CLICK_THRESH = 15
MARGIN       = 20  # pixels around extents
COLORS = {
    'default':  (200, 200, 200),  # light gray
    'hallway':  (0, 255, 255),    # yellow
    'elevator': (255, 0, 0),      # red
    'full':     (0, 255, 0),      # green
}
SEL_OUTLINE  = (0, 0, 255)        # blue
EDGE_COLOR   = (100, 100, 100)    # dark gray
EDGE_THICK   = 2
EDGE_SEL_OUT = (0, 165, 255)      # orange for edge-selection


def parse_args():
    p = argparse.ArgumentParser(description="JSON-based floorplan editor")
    p.add_argument("--json",  required=True, help="nodes/edges JSON path")
    p.add_argument("--image", help="optional background image path")
    p.add_argument("--scale", type=float, default=1.0, help="zoom factor")
    return p.parse_args()


def load_data(path):
    data = json.loads(path.read_text())
    data.setdefault('nodes', [])
    data.setdefault('edges', [])
    return data


def compute_offset(nodes, scale):
    xs = [n['x'] for n in nodes]
    ys = [n['y'] for n in nodes]
    if not xs or not ys:
        return 0,0, 800,600
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width  = int((max_x - min_x) * scale + 2 * MARGIN)
    height = int((max_y - min_y) * scale + 2 * MARGIN)
    off_x = -min_x + MARGIN/scale
    off_y = -min_y + MARGIN/scale
    return off_x, off_y, width, height


def id2pt(nodes, floor, scale, offset):
    off_x, off_y = offset
    pts = {}
    for n in nodes:
        if n.get('floor', 0) == floor:
            x = int((n['x'] + off_x) * scale)
            y = int((n['y'] + off_y) * scale)
            pts[n['id']] = (x, y)
    return pts


def nearest_node(pts, click_pt):
    cx, cy = click_pt
    best, bd = None, CLICK_THRESH * CLICK_THRESH
    for nid, (x, y) in pts.items():
        d = (x - cx)**2 + (y - cy)**2
        if d <= bd:
            best, bd = nid, d
    return best


def redraw(canvas, bg_img, data, floor, scale, offset, sel_id, mode, edge_sel):
    # background
    if bg_img is not None:
        canvas[:] = bg_img
    else:
        canvas[:] = 255
    # draw edges
    pts = id2pt(data['nodes'], floor, scale, offset)
    for u, v in data['edges']:
        if u in pts and v in pts:
            cv2.line(canvas, pts[u], pts[v], EDGE_COLOR, EDGE_THICK)
    # draw nodes
    for n in data['nodes']:
        if n.get('floor', 0) != floor: continue
        nid = n['id']
        x0, y0 = pts[nid]
        tags = n.get('tags', {})
        if tags.get('elevator'):
            color = COLORS['elevator']
        elif tags.get('full'):
            color = COLORS['full']
        elif tags.get('hallway'):
            color = COLORS['hallway']
        else:
            color = COLORS['default']
        cv2.circle(canvas, (x0, y0), NODE_RADIUS, color, -1)
        cv2.circle(canvas, (x0, y0), NODE_RADIUS, (0,0,0), 1)
        lbl = ''.join(k[0].upper() for k, v in tags.items() if v)
        if lbl:
            cv2.putText(canvas, lbl, (x0 + NODE_RADIUS, y0 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    # highlight selection or edge picks
    if mode == 'edge':
        for nid in edge_sel:
            if nid in pts:
                cv2.circle(canvas, pts[nid], NODE_RADIUS+4, EDGE_SEL_OUT, 2)
    elif sel_id and sel_id in pts:
        cv2.circle(canvas, pts[sel_id], NODE_RADIUS+4, SEL_OUTLINE, 2)
    # floor & mode indicator
    cv2.putText(canvas, f"Floor: {floor}    Mode: {mode}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)


def main():
    args      = parse_args()
    json_path = pathlib.Path(args.json)
    data      = load_data(json_path)
    scale     = args.scale

    # optional background image
    bg_img = None
    if args.image:
        img = cv2.imread(str(args.image))
        if img is None:
            print(f"Error: cannot load image {args.image}")
            return
        h, w = img.shape[:2]
        bg_img = cv2.resize(img, (int(w*scale), int(h*scale)))

    # compute canvas and offset
    offset_x, offset_y, width, height = compute_offset(data['nodes'], scale)
    offset = (offset_x, offset_y)
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    floors = sorted({n.get('floor', 0) for n in data['nodes']})
    if not floors:
        floors = [0]
    idx       = 0
    sel_id    = None
    mode      = 'tag'   # start in tag mode
    edge_sel  = []

    cv2.namedWindow('editor', cv2.WINDOW_NORMAL)
    def on_mouse(evt, x, y, flags, param):
        nonlocal sel_id, edge_sel
        if evt == cv2.EVENT_LBUTTONDOWN:
            pts = id2pt(data['nodes'], floors[idx], scale, offset)
            nid = nearest_node(pts, (x, y))
            if mode == 'edge':
                if nid and nid not in edge_sel:
                    edge_sel.append(nid)
                    if len(edge_sel) == 2:
                        data['edges'].append([edge_sel[0], edge_sel[1]])
                        print(f"Added edge: {edge_sel[0]} <-> {edge_sel[1]}")
                        edge_sel.clear()
            else:
                sel_id = nid
    cv2.setMouseCallback('editor', on_mouse)

    '''
    Controls: '['/']' switch floor | click select | e elevator | h hallway | f full | a edge mode | s save | q quit
    '''
    while True:
        redraw(canvas, bg_img, data, floors[idx], scale, offset, sel_id, mode, edge_sel)
        cv2.imshow('editor', canvas)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            break
        elif k == ord(']'):
            idx, sel_id = (idx + 1) % len(floors), None
            edge_sel.clear()
        elif k == ord('['):
            idx, sel_id = (idx - 1) % len(floors), None
            edge_sel.clear()
        elif k == ord('a'):
            mode = 'edge' if mode != 'edge' else 'tag'
            sel_id = None
            edge_sel.clear()
        elif k in (ord('e'), ord('h'), ord('f')) and mode != 'edge' and sel_id:
            node = next((n for n in data['nodes'] if n['id'] == sel_id), None)
            if node:
                key = {'e': 'elevator', 'h': 'hallway', 'f': 'full'}[chr(k)]
                tags = node.setdefault('tags', {'hallway': False, 'elevator': False, 'full': False})
                tags[key] = not tags[key]
                print(f"Toggled {key} on {sel_id} → {tags[key]}")
        elif k == ord('s'):
            json_path.write_text(json.dumps(data, indent=2))
            print(f"Saved {json_path}")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
