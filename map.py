import streamlit as st
import json, math
import cv2
import numpy as np
import pytesseract
from PIL import Image
from io import BytesIO
from skimage.morphology import skeletonize
from scipy.spatial import KDTree
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

st.set_page_config(layout="wide")
st.title("Floorplan Node & Edge Editor")

# ---- Sidebar: upload and preprocess ----
with st.sidebar:
    st.header("1. Upload & Preprocess")
    uploaded = st.file_uploader("Upload floorplan image (PNG/JPG)", type=["png","jpg","jpeg"])
    preprocess = st.button("Preprocess: OCR + Skeleton")
    st.markdown("---")
    st.header("2. Export")
    export_name = st.text_input("JSON filename", value="floor_graph.json")
    if st.button("Save JSON"):
        js = {"nodes": st.session_state.nodes, "edges": st.session_state.edges}
        st.download_button("Download JSON", data=json.dumps(js, indent=2), file_name=export_name)

# ---- Initialize state ----
st.session_state.setdefault('nodes', [])
st.session_state.setdefault('edges', [])
st.session_state.setdefault('image', None)
st.session_state.setdefault('img_size', (0,0))

# ---- Preprocess ----
if uploaded:
    # load image once
    img = Image.open(uploaded).convert("RGB")
    st.session_state.image = img
    w, h = img.size
    st.session_state.img_size = (w, h)
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    st.image(img, caption="Uploaded Floorplan", use_column_width=True)

    if preprocess:
        # OCR detection
        d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        for i, text in enumerate(d['text']):
            tid = text.strip()
            if not tid: continue
            x, y, w_box, h_box = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
            cx, cy = x + w_box/2, y + h_box/2
            if not any(n['id']==tid for n in st.session_state.nodes):
                st.session_state.nodes.append({'id':tid, 'x':float(cx), 'y':float(cy), 'floor':0})
        # Skeleton extraction
        _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        sk = skeletonize(bw//255)
        ys, xs = np.nonzero(sk)
        sk_pts = list(zip(xs.tolist(), ys.tolist()))
        # cluster junctions
        tree = KDTree(sk_pts)
        for x0, y0 in sk_pts:
            dists, idxs = tree.query((x0,y0), k=4)
            if np.sum(dists<2) > 2:
                jid = f"J{x0}_{y0}"
                if not any(n['id']==jid for n in st.session_state.nodes):
                    st.session_state.nodes.append({'id':jid, 'x':float(x0), 'y':float(y0), 'floor':0})
        # connect nearby nodes
        for i, n1 in enumerate(st.session_state.nodes):
            for j, n2 in enumerate(st.session_state.nodes[i+1:], start=i+1):
                dx = n1['x'] - n2['x']; dy = n1['y'] - n2['y']
                if math.hypot(dx,dy) < 50:
                    e = {'from':n1['id'], 'to':n2['id']}
                    if e not in st.session_state.edges:
                        st.session_state.edges.append(e)
        st.success("Preprocessing complete.")

# ---- Editor Canvas ----
st.header("3. Manual Edit")
cols = st.columns([3,1])
with cols[0]:
    if st.session_state.image:
        img = st.session_state.image
        w, h = st.session_state.img_size
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_layout_image(
            source=img,
            xref="x", yref="y",
            x=0, y=0,
            sizex=w, sizey=h,
            sizing="stretch",
            opacity=0.6)
        # nodes
        xs = [n['x'] for n in st.session_state.nodes]
        ys = [h - n['y'] for n in st.session_state.nodes]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers+text',
                                 text=[n['id'] for n in st.session_state.nodes],
                                 textposition='top center', marker=dict(size=6, color='red')))
        # edges
        for e in st.session_state.edges:
            a = next(n for n in st.session_state.nodes if n['id']==e['from'])
            b = next(n for n in st.session_state.nodes if n['id']==e['to'])
            fig.add_trace(go.Scatter(x=[a['x'], b['x']], y=[h-a['y'], h-b['y']],
                                     mode='lines', line=dict(color='yellow')))
        fig.update_xaxes(showgrid=False, visible=False)
        fig.update_yaxes(showgrid=False, visible=False, scaleanchor="x")
        fig.update_layout(width=800, height=int(800*h/w))
        st.plotly_chart(fig)
with cols[1]:
    st.write("**Add / Remove**")
    mode = st.radio("Mode", ['Add Node','Add Edge','Delete Node','Delete Edge'])
    if mode=='Add Node':
        nid = st.text_input('Node ID')
        floor = st.number_input('Floor', value=0)
        coord_in = st.text_input('x,y')
        if st.button('Add') and nid and coord_in:
            x,y = map(float, coord_in.split(','))
            st.session_state.nodes.append({'id':nid,'x':x,'y':y,'floor':floor})
    elif mode=='Add Edge':
        frm = st.selectbox('From', [n['id'] for n in st.session_state.nodes])
        to  = st.selectbox('To',   [n['id'] for n in st.session_state.nodes])
        if st.button('Add Edge'):
            st.session_state.edges.append({'from':frm,'to':to})
    elif mode=='Delete Node':
        did = st.selectbox('Node', [n['id'] for n in st.session_state.nodes])
        if st.button('Delete Node'):
            st.session_state.nodes = [n for n in st.session_state.nodes if n['id']!=did]
            st.session_state.edges = [e for e in st.session_state.edges if e['from']!=did and e['to']!=did]
    elif mode=='Delete Edge':
        labels = [f"{e['from']}→{e['to']}" for e in st.session_state.edges]
        sel = st.selectbox('Edge', labels)
        if st.button('Delete Edge') and sel:
            idx = labels.index(sel)
            st.session_state.edges.pop(idx)
