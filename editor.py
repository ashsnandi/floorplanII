#!/usr/bin/env python3
"""
floorplan_editor.py — interactive floorplan node/edge editor

Features:
  • Normal, hallway, branch, edge, and tag modes
  • Auto-chaining of hallway nodes
  • Branch mode: link new node to nearest hallway
  • Tag mode: click node then press 1/2/3 to toggle tags
  • Edges survive save/load cycles
  • Display scale, colored nodes, thick edges
  • c key continues hallway chain

Controls:
  n: normal (add/remove)
  h: hallway (auto-chain)
  c: continue-hallway
  b: branch (auto-link to hallway)
  e: edge (click two nodes)
  t: tag (select node, then 1=hallway,2=elevator,3=full)
  s: save
  q: quit
"""
import cv2
import json
import pathlib
import argparse

# Appearance constants
NODE_RADIUS  = 8
CLICK_THRESH = 15
NODE_COLOR   = (0, 255, 255)
NODE_OUTLINE = (0, 0, 0)
SEL_OUTLINE  = (0, 0, 255)
EDGE_COLOR   = (50, 150, 255)
EDGE_THICK   = 3
TAG_COLOR    = (255, 0, 0)


def parse_args():
    p = argparse.ArgumentParser(description="Interactive floorplan editor")
    p.add_argument("--image", required=True, help="floorplan image path")
    p.add_argument("--json",  required=True, help="nodes/edges JSON path")
    p.add_argument("--scale", type=float, default=1.0, help="UI zoom factor")
    return p.parse_args()


def main():
    args = parse_args()
    img_path  = pathlib.Path(args.image)
    json_path = pathlib.Path(args.json)
    scale     = args.scale

    bg = cv2.imread(str(img_path))
    if bg is None:
        print(f"Error: cannot load image {img_path}")
        return
    disp = cv2.resize(bg, None, fx=scale, fy=scale) if scale!=1.0 else bg.copy()
    canvas = disp.copy()

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
        except Exception as e:
            print(f"Error reading JSON: {e}")
            return
    else:
        data = {"nodes": [], "edges": []}
    data.setdefault("nodes", [])
    data.setdefault("edges", [])
    data["edges"] = [list(e) for e in data["edges"]]

    def id2pt():
        return {n['id']:(int(n['x']*scale), int(n['y']*scale)) for n in data['nodes']}

    def nearest(pt_disp):
        xd, yd = pt_disp
        for n in data['nodes']:
            dx = n['x']*scale - xd; dy = n['y']*scale - yd
            if dx*dx+dy*dy <= CLICK_THRESH*CLICK_THRESH:
                return n
        return None

    def nearest_hallway(pt_disp):
        xd, yd = pt_disp
        best, bd = None, float('inf')
        for n in data['nodes']:
            if n.get('tags',{}).get('hallway'):
                x,y = n['x']*scale, n['y']*scale
                d=(x-xd)**2+(y-yd)**2
                if d<bd:
                    best, bd = n, d
        return best

    # redraw function
    def redraw():
        nonlocal canvas
        canvas = disp.copy()
        pts = id2pt()
        # draw nodes
        for n in data['nodes']:
            x,y = pts[n['id']]
            cv2.circle(canvas,(x,y),NODE_RADIUS,NODE_COLOR,-1)
            cv2.circle(canvas,(x,y),NODE_RADIUS,NODE_OUTLINE,1)
            tags = n.get('tags',{})
            lbl = ''.join(k[0].upper() for k,v in tags.items() if v)
            if lbl:
                cv2.putText(canvas,lbl,(x+NODE_RADIUS+2,y-2),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,TAG_COLOR,1)
        # draw edges
        for u,v in data['edges']:
            if u in pts and v in pts:
                cv2.line(canvas,pts[u],pts[v],EDGE_COLOR,EDGE_THICK)
        # highlight edge-mode
        if mode=='edge':
            for sid in edge_sel:
                if sid in pts:
                    cv2.circle(canvas,pts[sid],NODE_RADIUS+4,SEL_OUTLINE,2)
        # highlight selected tag node
        if mode=='tag' and tag_target:
            if tag_target in pts:
                cv2.circle(canvas,pts[tag_target],NODE_RADIUS+6,SEL_OUTLINE,2)

    mode='normal'
    edge_sel=[]
    last_hallway=None
    tag_target=None

    def on_click(event,x_disp,y_disp,flags,param):
        nonlocal last_hallway, tag_target
        ox,oy = x_disp/scale, y_disp/scale
        if event==cv2.EVENT_LBUTTONDOWN:
            if mode=='normal':
                nid=f"N{len(data['nodes'])}_{int(ox)}_{int(oy)}"
                data['nodes'].append({'id':nid,'x':ox,'y':oy,'floor':0,'tags':{'hallway':False,'elevator':False,'full':False}})
                tag_target=None
            elif mode=='hallway':
                nid=f"N{len(data['nodes'])}_{int(ox)}_{int(oy)}"
                data['nodes'].append({'id':nid,'x':ox,'y':oy,'floor':0,'tags':{'hallway':True,'elevator':False,'full':False}})
                if last_hallway and last_hallway!=nid:
                    data['edges'].append([last_hallway,nid])
                last_hallway=nid
                tag_target=None
            elif mode=='branch':
                nid=f"N{len(data['nodes'])}_{int(ox)}_{int(oy)}"
                data['nodes'].append({'id':nid,'x':ox,'y':oy,'floor':0,'tags':{'hallway':False,'elevator':False,'full':False}})
                h=nearest_hallway((x_disp,y_disp))
                if h: data['edges'].append([nid,h['id']])
                tag_target=None
            elif mode=='edge':
                n=nearest((x_disp,y_disp))
                if n and n['id'] not in edge_sel:
                    edge_sel.append(n['id'])
                    if len(edge_sel)==2:
                        data['edges'].append(edge_sel.copy()); edge_sel.clear()
                tag_target=None
            elif mode=='tag':
                n=nearest((x_disp,y_disp))
                tag_target=n['id'] if n else None
        elif event==cv2.EVENT_RBUTTONDOWN and mode in('normal','hallway','branch'):
            n=nearest((x_disp,y_disp))
            if n:
                data['nodes'].remove(n)
                data['edges']=[(u,v) for u,v in data['edges'] if u!=n['id']and v!=n['id']]
                if mode=='hallway' and n['id']==last_hallway: last_hallway=None
                if mode=='tag' and n['id']==tag_target: tag_target=None
        redraw()

    # setup window
    cv2.namedWindow('editor',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('editor',on_click)
    redraw()

    print("Modes: n normal | h hallway | c continue | b branch | e edge | t tag | s save | q quit")
    print("In tag mode: click a node then press 1=hallway,2=elevator,3=full to toggle")
    while True:
        cv2.imshow('editor',canvas)
        k=cv2.waitKey(20)&0xFF
        if k==ord('q'): break
        elif k==ord('n'): mode,edge_sel,last_hallway,tag_target='normal',[],None,None
        elif k==ord('h'): mode='hallway'; tag_target=None
        elif k==ord('c'): mode='hallway'; tag_target=None
        elif k==ord('b'): mode='branch'; tag_target=None
        elif k==ord('e'): mode,edge_sel='edge',[]; tag_target=None
        elif k==ord('t'): mode,edge_sel='tag',[]; tag_target=None
        elif k in (ord('1'),ord('2'),ord('3')) and mode=='tag' and tag_target:
            n = next((x for x in data['nodes'] if x['id']==tag_target), None)
            if n:
                key = {'1':'hallway','2':'elevator','3':'full'}[chr(k)]
                tags=n.setdefault('tags',{'hallway':False,'elevator':False,'full':False})
                tags[key] = not tags[key]
                print(f"Toggled {key} on {tag_target} → {tags[key]}")
        elif k==ord('s'):
            pathlib.Path(json_path).write_text(json.dumps(data,indent=2))
            print(f"Saved {json_path}")
        redraw()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()