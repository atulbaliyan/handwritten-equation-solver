# Handwritten Equation Solver

A small Streamlit app that extracts handwritten or printed mathematical expressions from images and solves arithmetic expressions or single-variable equations.

**Features**
- Robust OCR pipeline combining template matching, Tesseract, and EasyOCR for better results on varied inputs.
- Algebra solver using SymPy for arithmetic evaluation and single-variable equation solving.
- Streamlit UI for quick uploads and interactive solving.

**Python Requirements**
- See `requirements.txt` for the Python packages used by the project.

**System Dependencies**
- Tesseract OCR (required by `pytesseract`):
  - On Debian/Ubuntu: `sudo apt update && sudo apt install -y tesseract-ocr`
  - On Fedora: `sudo dnf install -y tesseract`
  - On macOS (Homebrew): `brew install tesseract`
- If you plan to use GPU acceleration for `easyocr` / `torch`, install CUDA-enabled PyTorch per the official instructions at https://pytorch.org/get-started/locally/.

**Install (recommended)**
1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Note: Installing `torch`/`torchvision` via `pip` may require picking the correct wheel for your platform/CUDA configuration. If `pip install -r requirements.txt` fails for `torch`, follow the instructions on https://pytorch.org/get-started/locally/ and then re-run `pip install -r requirements.txt` to install the remaining packages.

**Run the App**

Start the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit in your browser, upload an image containing an equation, optionally edit the OCR result, and click "Solve".

**Quick CLI Tests**
- Test with a saved image (example scripts included):

```bash
python test_image.py
```

**Files**
- `ocr_engine.py`: OCR and preprocessing pipeline combining template matching, Tesseract, and EasyOCR.
- `solver.py`: Normalizes and solves arithmetic expressions and single-variable equations using SymPy.
- `streamlit_app.py`: Streamlit UI wiring the OCR and solver together.
- `test_image.py`: Simple test runners for local images.

**Troubleshooting**
- If `pytesseract` raises errors, ensure the `tesseract` binary is installed and on your `PATH`. You can set a custom path in code via `pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'`.
- If OCR results are poor, try adjusting image cropping or increasing contrast; the app already applies several preprocessing variants.
- If `easyocr` fails, ensure `torch` is installed with a compatible CPU/GPU wheel.

If you want, I can also create a small `README` section showing example input images and expected outputs, or add a `requirements-dev.txt` for development tools. Would you like that?
