#!/usr/bin/env python3
"""
floorplan_editor.py — automatic edge‑chaining in hallway mode

* Every time you add a hallway node, it automatically links to the
  previously added hallway node (same editing session).
* All other behaviour (edge‑mode, tag‑mode, etc.) unchanged.
* Switch out of hallway mode to stop auto‑chaining; switching back starts a
  new chain unless you press `c` to continue (see below).
"""
import cv2, json, pathlib, argparse

# ─── Appearance constants ──────────────────────────────────────────────────
NODE_RADIUS  = 8
CLICK_THRESH = 15
NODE_COLOR   = (0, 255, 255)
NODE_OUTLINE = (0, 0, 0)
SEL_OUTLINE  = (0, 0, 255)
EDGE_COLOR   = (50, 150, 255)
TAG_COLOR    = (255, 0, 0)
EDGE_THICK   = 3
# ───────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Floorplan node/edge editor")
    p.add_argument("--image", required=True)
    p.add_argument("--json",  required=True)
    p.add_argument("--scale", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    img_path, json_path, scale = map(pathlib.Path, [args.image, args.json, None])[:2] + [args.scale]

    bg = cv2.imread(str(img_path))
    if bg is None:
        print("Cannot load", img_path); return
    disp   = cv2.resize(bg, None, fx=scale, fy=scale) if scale != 1.0 else bg.copy()
    canvas = disp.copy()

    data = {"nodes": [], "edges": []}
    if json_path.exists():
        try:
            data.update(json.loads(json_path.read_text()))
        except Exception as e:
            print("Bad JSON:", e); return
    data.setdefault("nodes", [])
    data.setdefault("edges", [])

    def id2pt():
        return {n['id']: (int(n['x']*scale), int(n['y']*scale)) for n in data['nodes']}

    def redraw():
        nonlocal canvas
        canvas = disp.copy(); pts = id2pt()
        for u,v in data['edges']:
            if u in pts and v in pts:
                cv2.line(canvas, pts[u], pts[v], EDGE_COLOR, EDGE_THICK)
        for n in data['nodes']:
            x,y = pts[n['id']]
            cv2.circle(canvas,(x,y),NODE_RADIUS,NODE_COLOR,-1)
            cv2.circle(canvas,(x,y),NODE_RADIUS,NODE_OUTLINE,1)
            tags=n.get('tags',{})
            label=''.join(k[0].upper() for k,v in tags.items() if v)
            if label:
                cv2.putText(canvas,label,(x+NODE_RADIUS+2,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,TAG_COLOR,1)
        if mode=='edge':
            for sid in edge_sel:
                if sid in pts:
                    cv2.circle(canvas,pts[sid],NODE_RADIUS+4,SEL_OUTLINE,2)

    def nearest(pt_disp):
        xd,yd = pt_disp
        for n in data['nodes']:
            dx=n['x']*scale-xd; dy=n['y']*scale-yd
            if dx*dx+dy*dy <= CLICK_THRESH*CLICK_THRESH:
                return n
        return None

    mode='normal'; edge_sel=[]; last_hallway=None  # <- remembers previous hallway node id

    def on_click(ev,x_disp,y_disp,flags,param):
        nonlocal mode, edge_sel, last_hallway
        ox,oy = x_disp/scale, y_disp/scale
        if ev==cv2.EVENT_LBUTTONDOWN:
            if mode in ('normal','hallway'):
                nid=f"N{len(data['nodes'])}_{int(ox)}_{int(oy)}"
                node={'id':nid,'x':ox,'y':oy,'floor':0,'tags':{'hallway': mode=='hallway','elevator':False,'full':False}}
                data['nodes'].append(node)
                if mode=='hallway':
                    if last_hallway and last_hallway!=nid:
                        data['edges'].append((last_hallway,nid))
                    last_hallway = nid
            elif mode=='edge':
                n=nearest((x_disp,y_disp))
                if n and n['id'] not in edge_sel:
                    edge_sel.append(n['id'])
                    if len(edge_sel)==2:
                        data['edges'].append(tuple(edge_sel)); edge_sel=[]
            elif mode=='tag':
                n=nearest((x_disp,y_disp))
                if n:
                    key=input('Toggle tag (hallway/elevator/full): ')
                    tags=n.setdefault('tags',{'hallway':False,'elevator':False,'full':False})
                    if key in tags: tags[key]=not tags[key]
        elif ev==cv2.EVENT_RBUTTONDOWN and mode in ('normal','hallway'):
            n=nearest((x_disp,y_disp))
            if n:
                data['nodes'].remove(n)
                data['edges']=[(u,v) for u,v in data['edges'] if u!=n['id'] and v!=n['id']]
                if mode=='hallway' and n['id']==last_hallway:
                    last_hallway=None
        redraw()

    cv2.namedWindow('editor',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('editor',on_click)
    redraw()

    print('Modes: n normal | h hallway | c continue‑hallway | e edge | t tag | s save | q quit')
    while True:
        cv2.imshow('editor',canvas)
        k=cv2.waitKey(20)&0xFF
        if k==ord('q'): break
        elif k==ord('n'): mode='normal'; edge_sel=[]; last_hallway=None
        elif k==ord('h'): mode='hallway'; edge_sel=[]; last_hallway=None
        elif k==ord('c'): mode='hallway'; edge_sel=[]  # continue previous chain
        elif k==ord('e'): mode='edge'; edge_sel=[]
        elif k==ord('t'): mode='tag'; edge_sel=[]
        elif k==ord('s'):
            json_path.write_text(json.dumps(data,indent=2)); print('Saved',json_path)
        redraw()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
