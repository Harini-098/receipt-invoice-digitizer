#  IMPORTS 
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import pytesseract
import re
import sqlite3
import pandas as pd
import json
import matplotlib.pyplot as plt

# PAGE CONFIG 
st.set_page_config(page_title="Receipt Vault", layout="wide")

# SIDEBAR 
st.sidebar.header("🗄️ Vault Management")

if st.sidebar.button("🗑️ Clear All Records"):
    conn = sqlite3.connect("receipts.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM receipts")
    conn.commit()
    conn.close()
    st.sidebar.success("Vault reset successfully")

#  STYLES 
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.card {
    background-color: #111827;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

def card(title):
    st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)

#  IMAGE PREPROCESSING
def preprocess_image(pil_image):
    image = np.array(pil_image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #removes color

    clahe = cv2.createCLAHE(2.0, (8, 8)) #improves text contrast
    contrast = clahe.apply(gray)

    denoised = cv2.GaussianBlur(contrast, (3, 3), 0) #removes noise
    return denoised

#  OCR 
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\HARINI KAVETI\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

def extract_text(image):
    return pytesseract.image_to_string(image, config="--oem 3 --psm 6")

#  DATE & TIME EXTRACTION 
def extract_date_time(text):
    date = ""
    time = ""

    date_patterns = [
        r"\d{2}/\d{2}/\d{4}",
        r"\d{4}-\d{2}-\d{2}",
        r"\d{2}-\d{2}-\d{4}",
        r"\d{1,2}\s[A-Za-z]{3,9}\s\d{4}",
        r"\d{2}/\d{2}/\d{2}"
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            date = match.group()
            break

    time_patterns = [
        r"\d{1,2}:\d{2}\s?(AM|PM|am|pm)",
        r"\d{2}:\d{2}"
    ]

    for pattern in time_patterns:
        match = re.search(pattern, text)
        if match:
            time = match.group()
            break

    return date, time

# ================= FIELD EXTRACTION =================
def extract_fields(text):
    data = {
        "merchant": "",
        "date": "",
        "time": "",
        "total": "",
        "currency": ""
    }

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        data["merchant"] = lines[0]

    data["date"], data["time"] = extract_date_time(text)

    total_match = re.search(
        r"(total|amount|grand total)\s*[:\-]?\s*(\d+(\.\d{1,2})?)",
        text.lower()
    )
    if total_match:
        data["total"] = total_match.group(2)

    if "rs" in text.lower() or "₹" in text:
        data["currency"] = "INR"
    elif "$" in text:
        data["currency"] = "USD"

    return data

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect("receipts.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant TEXT,
            date TEXT,
            time TEXT,
            total TEXT,
            currency TEXT,
            raw_json TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def is_duplicate(merchant, date, total):
    conn = sqlite3.connect("receipts.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM receipts WHERE merchant=? AND date=? AND total=?",
        (merchant, date, total)
    )
    exists = cur.fetchone()
    conn.close()
    return exists is not None

def save_to_db(data):
    conn = sqlite3.connect("receipts.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO receipts (merchant, date, time, total, currency, raw_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data["merchant"],
        data["date"],
        data["time"],
        data["total"],
        data["currency"],
        json.dumps(data)
    ))
    conn.commit()
    conn.close()

# ================= UI =================
st.title("💾 Receipt & Invoice Digitizer")

tab1, tab2, tab3 = st.tabs([
    "📥 Upload & Process",
    "📊 Spending Dashboard",
    "🗂️ History Vault"
])

# ================= TAB 1 =================
with tab1:
    card("Multi-Format Upload & OCR")

    uploaded_file = st.file_uploader(
        "Upload JPG / PNG / PDF (multi-page supported)",
        type=["jpg", "jpeg", "png", "pdf"]
    )

    if uploaded_file:
        images = []

        if uploaded_file.type == "application/pdf":
            images = convert_from_bytes(uploaded_file.read(), dpi=300)
        else:
            images = [Image.open(uploaded_file).convert("RGB")]

        image = images[0]
        processed = preprocess_image(image)

        c1, c2 = st.columns(2)
        with c1:
            st.image(image, caption="Original Image", width=300)
        with c2:
            st.image(processed, caption="Cleaned Image", width=300)

        if st.button("🚀 Process & Save"):
            full_text = ""
            for img in images:
                full_text += extract_text(preprocess_image(img))

            data = extract_fields(full_text)

            if is_duplicate(data["merchant"], data["date"], data["total"]):
                st.warning("⚠️ Duplicate receipt detected")
            else:
                save_to_db(data)
                st.success("✅ Receipt saved successfully")
                st.json(data)


# ================= TAB 2 =================
with tab2:
    card("Spending Patterns Dashboard")

    conn = sqlite3.connect("receipts.db")
    df = pd.read_sql_query(
        "SELECT merchant, date, total FROM receipts",
        conn
    )
    conn.close()

    if df.empty:
        st.info("No data available.")
    else:
        # Clean data
        df["total"] = pd.to_numeric(df["total"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Prepare data
        df["month"] = df["date"].dt.to_period("M").astype(str)
        monthly_sum = df.groupby("month")["total"].sum()

        merchant_sum = (
            df.groupby("merchant")["total"]
            .sum()
            .sort_values(ascending=False)
        )

        # ---------- SIDE BY SIDE LAYOUT ----------
        col1, col2 = st.columns(2)

        # ===== LEFT: MONTHLY BAR GRAPH =====
        with col1:
            st.subheader("📅 Monthly Spending")
            st.bar_chart(monthly_sum, height=260)

        # ===== RIGHT: MERCHANT PIE CHART =====
        with col2:
            st.subheader("🏪 Merchant Expense")

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(
                merchant_sum,
                labels=merchant_sum.index,
                autopct="%1.1f%%",
                startangle=90
            )
            ax.axis("equal")

            st.pyplot(fig)


# ================= TAB 3 =================
with tab3:
    card("History Vault")

    conn = sqlite3.connect("receipts.db")
    df = pd.read_sql_query(
        "SELECT id, merchant, date, time, total, currency FROM receipts",
        conn
    )
    conn.close()

    if df.empty:
        st.info("No receipts stored yet.")
    else:
        # Convert date column
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # ---------- DATE FILTER (ADDED BACK ✅) ----------
        st.subheader("📅 Filter by Date")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")

        if start_date and end_date:
            df = df[
                (df["date"] >= pd.to_datetime(start_date)) &
                (df["date"] <= pd.to_datetime(end_date))
            ]

        # ---------- TABLE ----------
        st.dataframe(df, use_container_width=True)

        # ---------- DETAILS ----------
        selected_id = st.selectbox(
            "Select Receipt ID",
            df["id"].tolist()
        )

        conn = sqlite3.connect("receipts.db")
        row = pd.read_sql_query(
            "SELECT raw_json FROM receipts WHERE id=?",
            conn,
            params=(selected_id,)
        )
        conn.close()

        if not row.empty:
            st.subheader("🧾 Receipt Details")
            st.json(json.loads(row.iloc[0]["raw_json"]))
