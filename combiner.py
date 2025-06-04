# align_floors.py
"""
Interactive Floor Alignment & Metadata Tool

Usage:
    python align_floors.py /path/to/floor_jsons/ [--output-dir /out] [--autosave]

Loads all files matching `floor*.json` in the specified directory,
plots them together, and provides controls to:
  - translate (dx, dy)
  - scale
  - rotate
  - reassign each node's numeric `floor` attribute
  - toggle visibility of each file independently

When you press the "Save All" button, the aligned coordinates and
new floor values are written to individual JSONs and a combined JSON.
"""
import sys
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button, TextBox

def load_floor_files(directory):
    floors = {}
    pattern = os.path.join(directory, 'floor*_aligned_aligned_aligned.json')
    for path in sorted(glob.glob(pattern)):
        file_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            data = json.load(f)
        floors[file_id] = {
            'path': path,
            'nodes': data.get('nodes', []),
            'edges': data.get('edges', [])
        }
    return floors

def compute_pivots(floors):
    pivots = {}
    for fid, info in floors.items():
        xs = [n['x'] for n in info['nodes']]
        ys = [n['y'] for n in info['nodes']]
        pivots[fid] = (np.mean(xs), np.mean(ys)) if xs and ys else (0, 0)
    return pivots

def transform_point(x, y, pivot, angle, scale, dx, dy):
    ox, oy = pivot
    x0, y0 = x - ox, y - oy
    theta = np.deg2rad(angle)
    x0, y0 = x0 * scale, y0 * scale
    xr = x0 * np.cos(theta) - y0 * np.sin(theta)
    yr = x0 * np.sin(theta) + y0 * np.cos(theta)
    return xr + ox + dx, yr + oy + dy

def main(directory, output_dir=None, autosave=False):
    floors = load_floor_files(directory)
    pivots = compute_pivots(floors)
    transforms = {fid: {'dx':0,'dy':0,'scale':1,'angle':0,'new_floor':None,'visible':True}
                  for fid in floors}
    for fid in transforms:
        digits = ''.join(c for c in fid if c.isdigit())
        transforms[fid]['new_floor'] = int(digits) if digits else None

    current = list(floors.keys())[0]

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.25, bottom=0.30, right=0.95, top=0.95)

    # Visibility toggles
    ax_check = plt.axes([0.02, 0.55, 0.20, 0.35])
    labels = list(floors.keys())
    actives = [transforms[fid]['visible'] for fid in labels]
    check = CheckButtons(ax_check, labels, actives)

    def plot_all():
        ax.clear()
        for fid, info in floors.items():
            tf = transforms[fid]
            if not tf['visible']:
                continue
            xs, ys = [], []
            for node in info['nodes']:
                x2, y2 = transform_point(
                    node['x'], node['y'], pivots[fid],
                    tf['angle'], tf['scale'], tf['dx'], tf['dy']
                )
                xs.append(x2)
                ys.append(y2)
            label = f"{fid} → floor {tf['new_floor']}"
            ax.scatter(xs, ys, s=10, label=label)
        ax.invert_yaxis()
        ax.set_aspect('equal','box')
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        ax.set_title('Floor Alignment & Mapping')

    plot_all()

    # Sliders
    ax_dx = plt.axes([0.25, 0.25, 0.65, 0.03])
    slider_dx = Slider(ax_dx, 'dx', -2000, 2000, valinit=0)
    ax_dy = plt.axes([0.25, 0.20, 0.65, 0.03])
    slider_dy = Slider(ax_dy, 'dy', -2000, 2000, valinit=0)
    ax_scale = plt.axes([0.25, 0.15, 0.65, 0.03])
    slider_scale = Slider(ax_scale, 'scale', 0.1, 5, valinit=1)
    ax_ang = plt.axes([0.25, 0.10, 0.65, 0.03])
    slider_ang = Slider(ax_ang, 'angle', -180, 180, valinit=0)

    # New floor assignment
    ax_txt = plt.axes([0.02, 0.50, 0.20, 0.05])
    txt_floor = TextBox(ax_txt, 'Set floor →', initial=str(transforms[current]['new_floor']))

    # Save button
    ax_save = plt.axes([0.02, 0.44, 0.20, 0.05])
    btn_save = Button(ax_save, 'Save All')

    # Floor selector
    ax_radio = plt.axes([0.02, 0.02, 0.20, 0.40])  # taller, less squished
    radio = RadioButtons(ax_radio, labels, active=0)

    def on_visibility(label):
        transforms[label]['visible'] = not transforms[label]['visible']
        plot_all(); fig.canvas.draw_idle()

    def on_select(label):
        nonlocal current
        current = label
        tf = transforms[current]
        slider_dx.set_val(tf['dx'])
        slider_dy.set_val(tf['dy'])
        slider_scale.set_val(tf['scale'])
        slider_ang.set_val(tf['angle'])
        txt_floor.set_val(str(tf['new_floor']))

    def on_transform(val):
        tf = transforms[current]
        tf['dx'], tf['dy'] = slider_dx.val, slider_dy.val
        tf['scale'], tf['angle'] = slider_scale.val, slider_ang.val
        plot_all(); fig.canvas.draw_idle()

    def on_floor(text):
        try:
            transforms[current]['new_floor'] = int(text)
        except ValueError:
            pass
        plot_all(); fig.canvas.draw_idle()

    def on_save(event):
        out = output_dir or directory
        os.makedirs(out, exist_ok=True)
        combined = {'nodes': [], 'edges': []}
        for fid, info in floors.items():
            tf = transforms[fid]
            aligned_nodes = []
            for node in info['nodes']:
                x2, y2 = transform_point(
                    node['x'], node['y'], pivots[fid],
                    tf['angle'], tf['scale'], tf['dx'], tf['dy']
                )
                nn = node.copy()
                nn['x'], nn['y'] = x2, y2
                nn['floor'] = tf['new_floor']
                aligned_nodes.append(nn)
            fname = f"{fid}_aligned.json"
            with open(os.path.join(out, fname), 'w') as fo:
                json.dump({'nodes': aligned_nodes, 'edges': info['edges']}, fo, indent=2)
            combined['nodes'].extend(aligned_nodes)
            combined['edges'].extend(info['edges'])
        with open(os.path.join(out, 'combined_aligned.json'), 'w') as fc:
            json.dump(combined, fc, indent=2)
        print(f"Saved {len(floors)} files and combined JSON to {out}")

    check.on_clicked(on_visibility)
    radio.on_clicked(on_select)
    slider_dx.on_changed(on_transform)
    slider_dy.on_changed(on_transform)
    slider_scale.on_changed(on_transform)
    slider_ang.on_changed(on_transform)
    txt_floor.on_submit(on_floor)
    btn_save.on_clicked(on_save)

    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='Input directory')
    parser.add_argument('--out', dest='outdir', help='Output directory')
    parser.add_argument('--autosave', action='store_true')
    args = parser.parse_args()
    main(args.indir, args.outdir, args.autosave)
