# edit_edges_matplotlib.py
"""
Interactive 3-D Edge Editor with Matplotlib

 * Left-click selects a node (highlighted in blue).  
 * After two picks, draws a black edge connecting them.  
 * Press 's' to save all edges back to the JSON.  
 * Press 'c' to clear current selection.  

Requirements:
   pip install matplotlib numpy

Usage:
   python edit_edges_matplotlib.py path/to/mghTrue_chained.json
"""
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load JSON data
try:
    json_path = sys.argv[1]
except IndexError:
    print("Usage: python edit_edges_matplotlib.py <path/to/file.json>")
    sys.exit(1)
with open(json_path) as f:
    data = json.load(f)

nodes = data.get('nodes', [])
edges = set(tuple(e) for e in data.get('edges', []))
elev_id = {n['id']: n for n in nodes}

# Prepare coordinates
xs = np.array([n['x'] for n in nodes])
ys = np.array([n['y'] for n in nodes])
zs = np.array([n['floor'] for n in nodes])

# Matplotlib figure setup
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
pts = ax.scatter(xs, ys, zs, s=5, c='lightgray', picker=5)
# Map scatter point index to node ID
index2id = {i: nodes[i]['id'] for i in range(len(nodes))}
selected = []

# Redraw everything
def redraw():
    ax.cla()  # clear all artists
    # Re-plot all nodes\    
    ax.scatter(xs, ys, zs, s=5, c='lightgray')
    # Highlight selected nodes
    for sid in selected:
        n = elev_id[sid]
        ax.scatter([n['x']], [n['y']], [n['floor']], s=80,
                   facecolors='none', edgecolors='blue', linewidths=2)
    # Draw all edges
    for id1, id2 in edges:
        n1 = elev_id[id1]
        n2 = elev_id[id2]
        ax.plot([n1['x'], n2['x']], [n1['y'], n2['y']],
                [n1['floor'], n2['floor']], c='black', linewidth=2)
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Floor')
    ax.set_title("3-D Edge Editor  (click two nodes to link, 's'=save, 'c'=clear)")
    # Enable navigation toolbar for pan/zoom
    plt.tight_layout()
    fig.canvas.draw_idle()

# Handle pick events
def on_pick(event):
    ind = event.ind[0]
    nid = index2id[ind]
    selected.append(nid)
    if len(selected) == 2:
        edge = tuple(sorted(selected))
        if edge not in edges:
            edges.add(edge)
            print(f"Added edge: {edge}")
        selected.clear()
    redraw()
x
# Handle key presses
def on_key(event):
    if event.key == 's':
        data['edges'] = [list(e) for e in edges]
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(edges)} edges to {json_path}")
    elif event.key == 'c':
        selected.clear()
        redraw()

# Connect events
fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('key_press_event', on_key)

# Initial draw
redraw()
plt.show()