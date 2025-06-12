import streamlit as st
st.set_page_config(layout="wide", page_title="Crack Detector")
from main import (
    model,
    common_transform,
    crop_tiles,
    predict_cropped_images,
    generate_positive_mask,
    detect_cracks_within_mask,
    crack_edges_to_heatmap,
    overlay_heatmap_on_image
)
import cv2
import numpy as np

st.title("🧠Crackmap : Crack Detection using CNN + Edge Analysis + Heatmap Visualization")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
d = st.slider("Tile Division Factor (d)", 1, 10, 3)
min_line_length = st.slider("Minimum Crack Length (px)", 10, 500, 150)

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    # Step-by-step processing
    shape = img_rgb.shape[:2]
    cropped = crop_tiles(img_rgb, tile_size=(shape[0]//d, shape[1]//d))
    preds = predict_cropped_images(model, cropped, common_transform)
    mask = generate_positive_mask(preds["object_oriented"], shape)
    crack_edges = detect_cracks_within_mask(img_rgb, mask, min_length=min_line_length)
    crack_heatmap = crack_edges_to_heatmap(crack_edges)
    overlay = overlay_heatmap_on_image(img_rgb, crack_heatmap)

    # Show results
    st.subheader("🔍 Crack Detection Results")
    col1, col2 = st.columns(2)
    col1.image(mask, caption="Positive Region Mask", use_container_width=True)
    col2.image(crack_edges, caption="Crack Edges", use_container_width=True)

    st.image(overlay, caption="Final Overlay", use_container_width=True)
