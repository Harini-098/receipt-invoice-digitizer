# ================= IMPORTS =================
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import pytesseract

# ================= PAGE SETUP =================
st.set_page_config(page_title="Milestone 1 - OCR", layout="centered")
st.title("📄 Receipt OCR - Milestone 1")

# ================= TESSERACT PATH =================
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\HARINI KAVETI\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

# ================= IMAGE PREPROCESS =================
def preprocess_image(img):
    img = np.array(img)

    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Improve contrast
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(gray)

    # Reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return blurred

# ================= OCR FUNCTION =================
def extract_text(img):
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6")

# ================= FILE UPLOAD =================
file = st.file_uploader("Upload receipt (JPG, PNG, PDF)", ["jpg", "png", "pdf"])

if file:
    # Handle PDF or Image
    if file.type == "application/pdf":
        images = convert_from_bytes(file.read(), 300)
        image = images[0]
    else:
        image = Image.open(file)

    st.subheader("🖼️ Original Image")
    st.image(image, width=300)

    processed = preprocess_image(image)

    st.subheader("🧪 Processed Image")
    st.image(processed, width=300)

    if st.button("🔍 Extract Text"):
        text = extract_text(processed)

        st.subheader("📄 Extracted Raw Text")
        st.text_area("OCR Output", text, height=300)
