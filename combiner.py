# align_floors.py
"""
Interactive Floor Alignment & Metadata Tool

Usage:
    python align_floors.py /path/to/floor_jsons/ [--output-dir /out] [--autosave]

Loads all `floor*.json` files from the specified directory,
plots them together, and provides controls to:
  - translate (dx, dy)
  - scale
  - rotate
  - reassign the `floor` attribute for each JSON

When you press the "Save" button, the aligned coordinates and
new floor values are written to JSON files in the output directory.
"""
import sys
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox

def load_floors(directory):
    file_paths = sorted(glob.glob(os.path.join(directory, 'floor*.json')))
    floors = {}
    for path in file_paths:
        fname = os.path.basename(path)
        num_str = ''.join(c for c in fname if c.isdigit())
        floor_num = int(num_str.split('.')[0])
        with open(path) as f:
            data = json.load(f)
        floors[floor_num] = {
            'path': path,
            'nodes': data.get('nodes', []),
            'edges': data.get('edges', [])
        }
    return floors

def compute_pivots(floors):
    pivots = {}
    for floor, info in floors.items():
        xs = [n['x'] for n in info['nodes']]
        ys = [n['y'] for n in info['nodes']]
        pivots[floor] = (np.mean(xs), np.mean(ys)) if xs and ys else (0, 0)
    return pivots

def rotate_and_transform(x, y, pivot, angle, scale, dx, dy):
    ox, oy = pivot
    x0, y0 = x - ox, y - oy
    theta = np.deg2rad(angle)
    x0, y0 = x0 * scale, y0 * scale
    xr = x0 * np.cos(theta) - y0 * np.sin(theta)
    yr = x0 * np.sin(theta) + y0 * np.cos(theta)
    return xr + ox + dx, yr + oy + dy

def main(directory, output_dir=None, autosave=False):
    floors = load_floors(directory)
    pivots = compute_pivots(floors)
    transforms = {f: {'dx':0,'dy':0,'scale':1,'angle':0,'new_floor':f} for f in floors}
    current = sorted(floors.keys())[0]

    fig, ax = plt.subplots(figsize=(8,6))
    plt.subplots_adjust(left=0.25, bottom=0.4)

    def plot_all():
        ax.clear()
        for f, info in floors.items():
            tf = transforms[f]
            xs, ys = [], []
            for n in info['nodes']:
                x2, y2 = rotate_and_transform(
                    n['x'], n['y'], pivots[f],
                    tf['angle'], tf['scale'], tf['dx'], tf['dy']
                )
                xs.append(x2)
                ys.append(y2)
            ax.scatter(xs, ys, s=5, label=f"Floor {f} → {tf['new_floor']}")
        ax.invert_yaxis()
        ax.set_aspect('equal','box')
        ax.legend(loc='upper right')
        ax.set_title('Alignment & Floor Mapping')

    plot_all()

    # Sliders
    ax_dx    = plt.axes([0.25,0.30,0.65,0.03])
    slider_dx    = Slider(ax_dx,'dx',-2000,2000,valinit=0)
    ax_dy    = plt.axes([0.25,0.25,0.65,0.03])
    slider_dy    = Slider(ax_dy,'dy',-2000,2000,valinit=0)
    ax_scale = plt.axes([0.25,0.20,0.65,0.03])
    slider_scale = Slider(ax_scale,'scale',0.1,5,valinit=1)
    ax_ang   = plt.axes([0.25,0.15,0.65,0.03])
    slider_ang   = Slider(ax_ang,'angle',-180,180,valinit=0)

    # Floor textbox
    ax_txt = plt.axes([0.025,0.25,0.15,0.05])
    txt_floor = TextBox(ax_txt,'Set floor', initial=str(current))

    # Save button
    ax_btn = plt.axes([0.025,0.15,0.15,0.05])
    btn_save = Button(ax_btn,'Save')

    # Radio buttons
    ax_radio = plt.axes([0.025,0.5,0.15,0.35])
    radio = RadioButtons(ax_radio, [str(f) for f in sorted(floors)], active=0)

    def update_sliders(label):
        nonlocal current
        current = int(label)
        tf = transforms[current]
        slider_dx.set_val(tf['dx'])
        slider_dy.set_val(tf['dy'])
        slider_scale.set_val(tf['scale'])
        slider_ang.set_val(tf['angle'])
        txt_floor.set_val(str(tf['new_floor']))

    def update_transform(val):
        tf = transforms[current]
        tf['dx']    = slider_dx.val
        tf['dy']    = slider_dy.val
        tf['scale'] = slider_scale.val
        tf['angle'] = slider_ang.val
        plot_all()
        fig.canvas.draw_idle()

    def update_floor(text):
        try:
            newf = int(text)
            transforms[current]['new_floor'] = newf
        except ValueError:
            pass
        plot_all()
        fig.canvas.draw_idle()

    def on_save(event):
        out = output_dir or directory
        os.makedirs(out, exist_ok=True)
        for f, info in floors.items():
            data = {'nodes': [], 'edges': info['edges']}
            tf = transforms[f]
            for n in info['nodes']:
                x2, y2 = rotate_and_transform(
                    n['x'], n['y'], pivots[f],
                    tf['angle'], tf['scale'], tf['dx'], tf['dy']
                )
                nn = n.copy()
                nn['x'] = x2
                nn['y'] = y2
                nn['floor'] = tf['new_floor']
                data['nodes'].append(nn)
            fname = f"floor{f}_aligned.json"
            with open(os.path.join(out, fname),'w') as fo:
                json.dump(data, fo, indent=2)
        print(f"Saved {len(floors)} files to {out}")

    radio.on_clicked(update_sliders)
    slider_dx.on_changed(update_transform)
    slider_dy.on_changed(update_transform)
    slider_scale.on_changed(update_transform)
    slider_ang.on_changed(update_transform)
    txt_floor.on_submit(update_floor)
    btn_save.on_clicked(on_save)

    plt.show()

if __name__=='__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('indir', help='Input directory')
    p.add_argument('--out', dest='outdir', help='Output directory')
    p.add_argument('--autosave', action='store_true')
    args = p.parse_args()
    main(args.indir, args.outdir, args.autosave)
