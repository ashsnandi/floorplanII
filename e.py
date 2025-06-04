#!/usr/bin/env python3
"""
floorplan_json_editor.py — interactive JSON-based floorplan editor with group move

Features:
  • Load nodes/edges from JSON
  • Auto-scale canvas based on JSON extents
  • Switch between floors ([ and ])
  • Select nodes and toggle tags: e elevator, h hallway, f full
  • Edge mode: connect two nodes (a)
  • Group-select mode: toggle membership (g)
  • Move mode: drag selected group (m)
  • Visual highlighting and save back to JSON
Controls:
  '[' / ']' : switch floor
  left-click : select node / edge endpoint / group toggle / start move
  e          : toggle elevator tag
  h          : toggle hallway tag
  f          : toggle full tag
  a          : toggle edge mode
  g          : toggle group-select mode
  m          : toggle move mode (drag group)
  s          : save JSON
  q          : quit
"""
import cv2, json, pathlib, argparse, math
import numpy as np

# Appearance constants
NODE_RADIUS  = 8
CLICK_THRESH = 15
MARGIN       = 20
COLORS = {
    'default':  (200, 200, 200),
    'hallway':  (0, 255, 255),
    'elevator': (255, 0, 0),
    'full':     (0, 255, 0),
}
SEL_OUTLINE   = (0, 0, 255)
EDGE_COLOR    = (100, 100, 100)
EDGE_THICK    = 2
EDGE_SEL_OUT  = (0, 165, 255)
GROUP_OUTLINE = (128, 0, 128)

MODES = ['tag', 'edge', 'group', 'move']


def parse_args():
    p = argparse.ArgumentParser(description="JSON-based floorplan editor with group move")
    p.add_argument("--json", required=True, help="nodes/edges JSON path")
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
        return 0,0,800,600
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = int((max_x-min_x)*scale + 2*MARGIN)
    h = int((max_y-min_y)*scale + 2*MARGIN)
    off_x = -min_x + MARGIN/scale
    off_y = -min_y + MARGIN/scale
    return off_x, off_y, w, h


def id2pt(nodes, floor, scale, offset):
    off_x, off_y = offset
    pts = {}
    for n in nodes:
        if n.get('floor',0)==floor:
            x = int((n['x']+off_x)*scale)
            y = int((n['y']+off_y)*scale)
            pts[n['id']] = (x,y)
    return pts


def nearest_node(pts, click_pt):
    cx, cy = click_pt
    best, bd = None, CLICK_THRESH*CLICK_THRESH
    for nid,(x,y) in pts.items():
        d = (x-cx)**2 + (y-cy)**2
        if d <= bd:
            best, bd = nid, d
    return best


def redraw(canvas, data, floor, scale, offset, sel_id, mode, edge_sel, group_sel):
    canvas[:] = 255
    pts = id2pt(data['nodes'], floor, scale, offset)
    # draw edges
    for u,v in data['edges']:
        if u in pts and v in pts:
            cv2.line(canvas, pts[u], pts[v], EDGE_COLOR, EDGE_THICK)
    # draw nodes
    for n in data['nodes']:
        if n.get('floor',0)!=floor: continue
        nid = n['id']
        x0,y0 = pts[nid]
        tags = n.get('tags',{})
        if tags.get('elevator'): col = COLORS['elevator']
        elif tags.get('full'):     col = COLORS['full']
        elif tags.get('hallway'):  col = COLORS['hallway']
        else:                       col = COLORS['default']
        cv2.circle(canvas,(x0,y0),NODE_RADIUS,col,-1)
        cv2.circle(canvas,(x0,y0),NODE_RADIUS,(0,0,0),1)
        lbl = ''.join(k[0].upper() for k,v in tags.items() if v)
        if lbl:
            cv2.putText(canvas,lbl,(x0+NODE_RADIUS,y0-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    # highlights
    if mode=='edge':
        for nid in edge_sel:
            if nid in pts: cv2.circle(canvas,pts[nid],NODE_RADIUS+4,EDGE_SEL_OUT,2)
    elif mode in ('tag','group') and sel_id and sel_id in pts:
        cv2.circle(canvas,pts[sel_id],NODE_RADIUS+4,SEL_OUTLINE,2)
    if group_sel:
        for nid in group_sel:
            if nid in pts: cv2.circle(canvas,pts[nid],NODE_RADIUS+6,GROUP_OUTLINE,2)
    # mode/floor label
    cv2.putText(canvas,f"Floor: {floor}  Mode: {mode}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)


def main():
    args = parse_args()
    json_path = pathlib.Path(args.json)
    data = load_data(json_path)
    scale = args.scale

    off_x,off_y,w,h = compute_offset(data['nodes'],scale)
    offset = (off_x, off_y)
    canvas = np.ones((h,w,3),dtype=np.uint8)*255

    floors = sorted({n.get('floor',0) for n in data['nodes']}) or [0]
    idx = 0
    sel_id = None
    edge_sel = []
    group_sel = []
    mode = 'tag'
    dragging = False
    drag_start = (0,0)
    orig_positions = {}

    cv2.namedWindow('editor',cv2.WINDOW_NORMAL)
    def on_mouse(evt, x, y, flags, param):
        nonlocal sel_id, dragging, drag_start, orig_positions, data, edge_sel, group_sel, mode
        pts = id2pt(data['nodes'], floors[idx], scale, offset)
        if evt==cv2.EVENT_LBUTTONDOWN:
            nid = nearest_node(pts, (x,y))
            if mode=='edge':
                if nid and nid not in edge_sel:
                    edge_sel.append(nid)
                    if len(edge_sel)==2:
                        data['edges'].append(edge_sel.copy()); print(f"Edge {edge_sel} added"); edge_sel.clear()
            elif mode=='group':
                if nid:
                    if nid in group_sel: group_sel.remove(nid)
                    else: group_sel.append(nid)
                    print(f"Group selection now: {group_sel}")
            elif mode=='move':
                if nid and nid in group_sel:
                    dragging = True
                    drag_start = (x,y)
                    orig_positions = {nid_:(n['x'],n['y']) for nid_ in group_sel for n in data['nodes'] if n['id']==nid_}
            else:
                sel_id = nid
        elif evt==cv2.EVENT_MOUSEMOVE and mode=='move' and dragging:
            dx = (x - drag_start[0]) / scale
            dy = (y - drag_start[1]) / scale
            for nid_, (ox,oy) in orig_positions.items():
                for n in data['nodes']:
                    if n['id']==nid_:
                        n['x'] = ox + dx
                        n['y'] = oy + dy
        elif evt==cv2.EVENT_LBUTTONUP and mode=='move' and dragging:
            dragging = False
        redraw(canvas, data, floors[idx], scale, offset, sel_id, mode, edge_sel, group_sel)
    cv2.setMouseCallback('editor', on_mouse)

    print("Controls: [/] switch floor | e/h/f toggle tags | a edge mode | g group mode | m move mode | s save | q quit")
    while True:
        redraw(canvas, data, floors[idx], scale, offset, sel_id, mode, edge_sel, group_sel)
        cv2.imshow('editor',canvas)
        k = cv2.waitKey(20)&0xFF
        if k==ord('q'): break
        elif k==ord(']'): idx, sel_id = (idx+1)%len(floors), None; edge_sel.clear()
        elif k==ord('['): idx, sel_id = (idx-1)%len(floors), None; edge_sel.clear()
        elif k==ord('a'): mode = 'edge' if mode!='edge' else 'tag'; sel_id=None; edge_sel.clear()
        elif k==ord('g'): mode = 'group' if mode!='group' else 'tag'; sel_id=None; edge_sel.clear()
        elif k==ord('m'):
            if group_sel:
                mode = 'move' if mode!='move' else 'tag'; sel_id=None; edge_sel.clear()
            else:
                print("No nodes selected—use 'g' to group first.")
        elif k in (ord('e'),ord('h'),ord('f')) and mode=='tag' and sel_id:
            node = next((n for n in data['nodes'] if n['id']==sel_id),None)
            if node:
                key={'e':'elevator','h':'hallway','f':'full'}[chr(k)]
                tags=node.setdefault('tags',{'hallway':False,'elevator':False,'full':False})
                tags[key]=not tags[key]; print(f"Toggled {key} on {sel_id} → {tags[key]}")
        elif k==ord('s'):
            json_path.write_text(json.dumps(data,indent=2)); print(f"Saved {json_path}")
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
