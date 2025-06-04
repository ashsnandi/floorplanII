#!/usr/bin/env python3
"""
floorplan_json_editor.py — full‑JSON editor with floor‑level translate / scale

New in this version
===================
• **Floor‑transform mode (`p`)**
  – Press **`p`** to toggle   (think *Plane* mode).
  – Click‑drag anywhere to translate **every node on the current floor**.
  – Press **`=` / `-`** to scale the floor up / down (about its centroid) in 10 % steps.

Existing goodies
----------------
• Tag mode (default), edge mode (`a`), group mode (`g`), move group (`m`).
• Save (`s`), floor flip (`[` / `]`).

Quick cheatsheet
----------------
```
[ / ]   cycle floor
p       toggle floor‑transform mode
  ↳ drag = translate floor
  ↳ =     scale up  (110 %)
  ↳ -     scale down (90 %)
g       group select     (click nodes)
m       move selected    (drag)
a       edge mode        (pick two nodes)
 e/h/f  toggle tags on selected node (in tag mode)
s       save JSON
q       quit
```
"""
import cv2, json, pathlib, argparse, math
import numpy as np

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')  # force Qt to XCB

# ----------------- appearance -----------------
NODE_R  = 7
CLICK_R = 14
COLORS  = {
    'default': (180,180,180),
    'hallway': (0,255,255),
    'elevator':(255,0,0),
    'full':    (0,255,0),
}
COL_EDGE = (110,110,110)
COL_SEL  = (0,0,255)
COL_GRP  = (128,0,128)

# ----------------- helpers -----------------
def id2pt(nodes, floor, scale, offset):
    ox, oy = offset
    return {n['id']:(int((n['x']+ox)*scale), int((n['y']+oy)*scale))
            for n in nodes if n.get('floor',0)==floor}

def near(pts, p):
    x,y = p; best=None; bd=CLICK_R**2
    for nid,(px,py) in pts.items():
        d=(px-x)**2+(py-y)**2
        if d<bd: best,bd=nid,d
    return best

def centroid(nodes):
    xs=[n['x'] for n in nodes]; ys=[n['y'] for n in nodes]
    return (sum(xs)/len(xs), sum(ys)/len(ys)) if xs else (0,0)

# ----------------- main -----------------
parser=argparse.ArgumentParser(description="Floor‑plan JSON editor w/ floor transform")
parser.add_argument('--json',required=True)
parser.add_argument('--scale',type=float,default=1.0)
args=parser.parse_args()
path=pathlib.Path(args.json)
data=json.loads(path.read_text())
for k in ('nodes','edges'): data.setdefault(k,[])
scale=args.scale
# offset so everything positive
xs=[n['x'] for n in data['nodes']]; ys=[n['y'] for n in data['nodes']]
ox,oy = -min(xs)+20/scale, -min(ys)+20/scale
offset=(ox,oy)
# window
cv2.namedWindow('editor',cv2.WINDOW_NORMAL)

floors=sorted({n.get('floor',0) for n in data['nodes']}) or [0]
idx=0; mode='tag'; sel=None; grp=[]; edge_sel=[]; dragging=False
orig_pos={}; drag_start=(0,0)

def redraw():
    global canvas
    pts=id2pt(data['nodes'],floors[idx],scale,offset)
    h=max(py for _,py in pts.values())+40 if pts else 600
    w=max(px for px,_ in pts.values())+40 if pts else 800
    canvas=np.ones((h,w,3),np.uint8)*255
    # edges
    for u,v in data['edges']:
        if u in pts and v in pts:
            cv2.line(canvas,pts[u],pts[v],COL_EDGE,1)
    # nodes
    for n in data['nodes']:
        if n.get('floor',0)!=floors[idx]: continue
        x,y=pts[n['id']]
        tag=n.get('tags',{})
        col=(COLORS['elevator'] if tag.get('elevator') else
              COLORS['full']     if tag.get('full') else
              COLORS['hallway']  if tag.get('hallway') else COLORS['default'])
        cv2.circle(canvas,(x,y),NODE_R,col,-1)
        cv2.circle(canvas,(x,y),NODE_R,(0,0,0),1)
    # highlights
    if mode=='edge':
        for nid in edge_sel:
            if nid in pts: cv2.circle(canvas,pts[nid],NODE_R+3,(0,165,255),2)
    if sel and sel in pts:
        cv2.circle(canvas,pts[sel],NODE_R+4,COL_SEL,2)
    for nid in grp:
        if nid in pts: cv2.circle(canvas,pts[nid],NODE_R+5,COL_GRP,2)
    cv2.putText(canvas,f"Floor {floors[idx]}  Mode {mode}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

redraw()

def on_mouse(evt,x,y,flags,param):
    global sel,grp,edge_sel,dragging,drag_start,orig_pos
    pts=id2pt(data['nodes'],floors[idx],scale,offset)
    nid=near(pts,(x,y))
    if evt==cv2.EVENT_LBUTTONDOWN:
        if mode=='edge':
            if nid and nid not in edge_sel:
                edge_sel.append(nid)
                if len(edge_sel)==2:
                    data['edges'].append(edge_sel.copy()); edge_sel.clear()
        elif mode=='group':
            if nid:
                grp.remove(nid) if nid in grp else grp.append(nid)
        elif mode=='move':
            if nid and nid in grp:
                dragging=True; drag_start=(x,y)
                orig_pos={n['id']:(n['x'],n['y']) for n in data['nodes'] if n['id'] in grp}
        elif mode=='floor':
            dragging=True; drag_start=(x,y)
            orig_pos={n['id']:(n['x'],n['y']) for n in data['nodes'] if n.get('floor',0)==floors[idx]}
        else:
            sel=nid
    elif evt==cv2.EVENT_MOUSEMOVE and dragging:
        dx=(x-drag_start[0])/scale; dy=(y-drag_start[1])/scale
        for nid,(ox,oy) in orig_pos.items():
            node=next(n for n in data['nodes'] if n['id']==nid)
            node['x']=ox+dx; node['y']=oy+dy
        redraw()
    elif evt==cv2.EVENT_LBUTTONUP:
        dragging=False

cv2.setMouseCallback('editor',on_mouse)
print("See docstring for keys.")

while True:
    cv2.imshow('editor',canvas)
    k=cv2.waitKey(20)&0xFF
    if k==ord('q'): break
    elif k==ord(']'): idx=(idx+1)%len(floors); sel=None; redraw()
    elif k==ord('['): idx=(idx-1)%len(floors); sel=None; redraw()
    elif k==ord('a'): mode='edge' if mode!='edge' else 'tag'; sel=None; edge_sel.clear(); redraw()
    elif k==ord('g'): mode='group' if mode!='group' else 'tag'; sel=None; redraw()
    elif k==ord('m'): mode='move' if (grp and mode!='move') else 'tag'; sel=None; redraw()
    elif k==ord('p'): mode='floor' if mode!='floor' else 'tag'; sel=None; redraw()
    elif k in (ord('='),ord('-')) and mode=='floor':
        # scale current floor about its centroid
        fl_nodes=[n for n in data['nodes'] if n.get('floor',0)==floors[idx]]
        cx,cy=centroid(fl_nodes)
        s=1.1 if k==ord('=') else 0.9
        for n in fl_nodes:
            n['x']=cx+(n['x']-cx)*s
            n['y']=cy+(n['y']-cy)*s
        redraw()
    elif k in (ord('e'),ord('h'),ord('f')) and mode=='tag' and sel:
        node=next(n for n in data['nodes'] if n['id']==sel)
        key={'e':'elevator','h':'hallway','f':'full'}[chr(k)]
        node.setdefault('tags',{'hallway':False,'elevator':False,'full':False})
        node['tags'][key]=not node['tags'][key]
        redraw()
    elif k==ord('s'):
        path.write_text(json.dumps(data,indent=2)); print('Saved.')

cv2.destroyAllWindows()
