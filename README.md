
# CrackMap

CrackMap is a deep learning-based image analysis tool designed for detecting cracks in materials using computer vision. It includes preprocessing steps like tiling and normalization, and supports model inference via PyTorch.

## Features

- Crack detection using a trained deep learning model (e.g. ResNet18)
- Image preprocessing with tiling and transformations,CLAHE,skeleton,Heatmap, Gaussianblur, etc
- Streamlit interface for easy demo 

## Installation

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

**For Windows (CMD):**

```cmd
python -m venv venv
venv\Scripts\activate
```

**For Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

## Usage

You can run the main script (adjust according to your entry point):

```bash
python main.py
```

Or, if using Streamlit:

```bash
streamlit run app.py
```

## License

This project is licensed under the MIT License.


