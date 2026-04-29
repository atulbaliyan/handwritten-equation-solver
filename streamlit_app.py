import io

import streamlit as st
from PIL import Image

from ocr_engine import extract_best_expression
from solver import solve_math

st.set_page_config(page_title="Handwritten Equation Solver", layout="wide")
st.title("Handwritten Equation Solver")
st.caption("Robust OCR + algebra solver. Supports arithmetic and single-variable equations.")

uploaded = st.file_uploader("Upload equation image", type=["png", "jpg", "jpeg", "webp"])

col1, col2 = st.columns(2)

image = None
if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

with col1:
    if image is not None:
        st.image(image, caption="Input Image", use_container_width=True)

with col2:
    if image is not None:
        best, all_candidates = extract_best_expression(image)
        st.text_input("Best OCR", value=best, key="best_ocr")
        edited = st.text_input("Edit OCR (if needed)", value=best, key="edited")
        with st.expander("All OCR candidates"):
            st.write(all_candidates)

        if st.button("Solve", type="primary"):
            try:
                res = solve_math(edited)
                st.success("Solved")
                st.write(f"Mode: `{res.mode}`")
                st.write(f"Normalized: `{res.normalized}`")
                st.write("Solution(s):")
                for s in res.solutions:
                    st.write(f"- {s}")
            except Exception as exc:
                st.error(f"Could not solve: {exc}")
    else:
        st.info("Upload an image to begin")
