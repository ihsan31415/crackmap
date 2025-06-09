import streamlit as st
import torch
import torch.nn.functional as F
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from io import BytesIO
from tqdm import tqdm

# Streamlit settings
st.set_page_config(layout="wide", page_title="Crack Detector")

# ---- Transform ----
common_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---- Load model ----
@st.cache_resource
def load_model():
    num_classes = 2
    model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load("concrete_crack_model_new.pth", map_location=torch.device("cpu")))
    model.to("cpu").eval()
    return model

model = load_model()

# ---- Utility Functions ----
def crop_tiles(image_rgb, tile_size):
    tiles = []
    height, width, _ = image_rgb.shape
    tile_w, tile_h = tile_size

    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            end_x = min(x + tile_w, width)
            end_y = min(y + tile_h, height)
            tile = image_rgb[y:end_y, x:end_x]
            h_actual, w_actual, _ = tile.shape
            if w_actual < 5 or h_actual < 5:
                continue
            coords = (x, y, w_actual, h_actual)
            tiles.append((tile, coords))
    return {"object_oriented": tiles}

def predict_cropped_images(model, cropped_image_dict, transform):
    predictions = {}
    for category, images_with_coords in cropped_image_dict.items():
        predictions[category] = []
        for img_array, coords in images_with_coords:
            img_tensor = transform(Image.fromarray(img_array)).unsqueeze(0).to("cpu")
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence = probs[0, 1].item()
                prediction = "Positive" if confidence > 0.7 else "Negative"
                predictions[category].append((img_array, prediction, confidence, coords))
    return predictions

def generate_positive_mask(predictions_list, original_shape, factor=1.2, threshold=0.7):
    heatmap = np.zeros(original_shape, dtype=np.float32)
    for _, pred, conf, (x, y, w, h) in predictions_list:
        if pred == "Positive" and conf > threshold:
            scaled_w, scaled_h = int(w * factor * conf), int(h * factor * conf)
            cx, cy = x + w // 2, y + h // 2
            x1 = max(0, cx - scaled_w // 2)
            y1 = max(0, cy - scaled_h // 2)
            x2 = min(original_shape[1], cx + scaled_w // 2)
            y2 = min(original_shape[0], cy + scaled_h // 2)
            heatmap[y1:y2, x1:x2] += 1.0
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=30)
    _, binary_mask = cv2.threshold(heatmap, 50, 255, cv2.THRESH_BINARY)
    return binary_mask.astype(np.uint8)

def detect_cracks_within_mask(img_rgb, binary_mask, low=50, high=100, min_length=100):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    clahe = cv2.createCLAHE(3.0, (8, 8))
    clahe_img = clahe.apply(binary)
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    enhanced = (clahe_img.astype(np.float32) / 255.0) * dist
    enhanced = np.uint8(enhanced * 255)
    edges = cv2.Canny(cv2.GaussianBlur(enhanced, (3, 3), 0.5), low, high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge_clean = np.zeros_like(edges)
    for cnt in contours:
        if cv2.arcLength(cnt, False) >= min_length:
            cv2.drawContours(edge_clean, [cnt], -1, 255, 1)
    return edge_clean

def crack_edges_to_heatmap(crack_edges, intensity=10, gamma=0.25):
    amplified = np.clip(crack_edges.astype(np.float32) * intensity, 0, 255)
    heatmap = cv2.GaussianBlur(amplified, (0, 0), sigmaX=15)
    heatmap = np.power(heatmap / 255.0, gamma) * 255.0
    norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

def overlay_heatmap_on_image(rgb, heatmap, alpha=0.6):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (bgr.shape[1], bgr.shape[0]))
    return cv2.cvtColor(cv2.addWeighted(heatmap, alpha, bgr, 1 - alpha, 0), cv2.COLOR_BGR2RGB)

# ---- Streamlit UI ----
st.title("🧠 Crack Detection using CNN + Edge Analysis")

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
