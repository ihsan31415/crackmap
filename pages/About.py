import streamlit as st

st.title("📄 About This App")

st.markdown("""
This is a crack detection app designed for visual inspection of concrete or asphalt surfaces.

It works by:
1. **Tiling the input image** based on the `d` value (divides image into smaller patches),
2. **Classifying each tile** using a pretrained ResNet18 model to detect possible cracks,
3. **Generating a heatmap** from confident crack regions,
4. **Enhancing the image** using CLAHE and edge detection,
5. **Drawing crack boundaries** and overlaying them back on the image.

### Parameters
- **Tile Division (`d`)**: Controls the granularity of image partitioning.
- **Minimum Crack Length**: Filters out short or noisy detections.

Built using:
- PyTorch + TIMM for classification
- OpenCV for enhancement
- Streamlit for web interface


Developed by: **Khoirul Ihsan**
""")

st.markdown("""
    Here's the step-by-step pipeline of **CrackNet**, from tiling to output visualization:

    ```
    Input Image
         ⬇️
    Image Tiling (Patch Extraction)
         ⬇️
    Tile Preprocessing & Normalization
         ⬇️
    Tile Classification (Crack Detection using CrackNet)
         ⬇️
    Positive Region Aggregation (Binary Mask Generation)
         ⬇️
    Contrast Enhancement & Edge Detection
         ⬇️
    Morphological Thinning (Skeletonization)
         ⬇️
    Heatmap Generation (Edge-to-Heatmap Conversion)
         ⬇️
    Heatmap Overlay on Original Image
         ⬇️
    Output Visualization
    ```

    **Detailed Steps:**

    1. **Input Image**  
       Load and prepare the input image for processing.

    2. **Image Tiling (Patch Extraction)**  
       The image is split into smaller tiles to allow localized crack detection.

    3. **Tile Preprocessing & Normalization**  
       Tiles are resized, normalized, and converted to tensors suitable for the model.

    4. **Tile Classification (Crack Detection using CrackNet)**  
       Each tile is classified to determine if it contains cracks.

    5. **Positive Region Aggregation (Binary Mask Generation)**  
       Tiles predicted positive are aggregated into a binary mask highlighting crack regions.

    6. **Contrast Enhancement & Edge Detection**  
       CLAHE and Canny edge detection are applied within masked regions to enhance crack edges.

    7. **Morphological Thinning (Skeletonization)**  
       Edges are thinned to a single-pixel skeleton to accurately represent cracks.

    8. **Heatmap Generation (Edge-to-Heatmap Conversion)**  
       Crack skeletons are converted into a smooth, colorful heatmap.

    9. **Heatmap Overlay on Original Image**  
       The heatmap is blended with the original image for visualization.

    10. **Output Visualization**  
        Final image with highlighted cracks is displayed for analysis.
    """
)