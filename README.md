# CrackMap

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-EE4C2C?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-5C3EE8?logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)

CrackMap is a deep learning-powered image analysis tool designed for detecting cracks in materials using computer vision. It provides a complete pipeline from image preprocessing (tiling, enhancement, transformation) to crack inference using a pretrained model (e.g., ResNet18). A Streamlit interface is available for an easy-to-use demo.

---

## 🔧 Features

- ✅ Crack detection using a PyTorch deep learning model (e.g., ResNet18)
- 🧩 Image preprocessing:
  - Tiling large images
  - CLAHE (adaptive histogram equalization)
  - Gaussian Blur
  - Skeletonization
  - Normalization & tensor transformation
- 📊 Visualization:
  - Heatmap overlays
  - Cracked region masks
- 🖼️ Streamlit-based web interface for real-time interaction

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ihsan31415/crackmap.git
cd crackmap
````

### 2. Create and Activate Virtual Environment

**For Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage
### Streamlit Web App
```bash
streamlit run app.py
```
Optional arguments:

* `Tile Division Factor (d)`: Size of each tile (usually 4, ex. = 4x4)
* `Minimum Crack Length (px)`: minimum lenght of the crack line (too smaal, too sensitive)

Upload an image through the UI, visualize Heatmap.

![Alt Text](assets/page.jpg)

---

## 🧠 Model Details

* Backbone: ResNet18
* Framework: PyTorch
* Input size: 224x224 

To train your own model, refer to `notebook.ipynb`.

---

## 📁 Folder Structure

```
crackmap/
├── app.py                               # streamlit page
├── main.py                              # main app
├── models/                              # Saved PyTorch models
|   └── concrete_crack_model_new.pth             
└── pages/
|   └── About.py
├── requirements.txt
├── assets/
├── notebook.ipynb
└── README.md

```

---

## 🧪 Supported Image Formats

* `.jpg`, `.jpeg`
* `.png`

---

## 📷 Sample Results
![Alt Text](assets/input.jpg)
![Alt Text](assets/results.jpg)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork and submit a pull request.

### 👇 Ways you can contribute:
- ⭐ Star this repo to show your support
- 🐛 Open an issue if you find a bug or have a suggestion
- 📥 Submit a pull request if you've made improvements
- 📣 Share it with others who might find it useful

If you're interested in collaborating or just want to say hi, feel free to connect!

📧 Email: [ihsanmuhammadkhoirul@gmail.com](mailto:ihsanmuhammadkhoirul@gmail.com)  
🔗 LinkedIn: [khoirul ihsan](https://www.linkedin.com/in/khoirul-ihsan-387115288/)

---

## 🌐 Credits

Built by [ihsan31415](https://github.com/ihsan31415), pls contrib if u want <3.
Inspired by real-world needs in infrastructure and material health monitoring.

