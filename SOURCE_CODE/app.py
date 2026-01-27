# ================= IMPORTS =================
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
import uuid

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Receipt Vault", layout="wide")

# ================= SIDEBAR =================
st.sidebar.header("🗄️ Vault Management")

if st.sidebar.button("🗑️ Clear All Records"):
    conn = sqlite3.connect("receipts.db")
    conn.execute("DELETE FROM receipts")
    conn.commit()
    conn.close()
    st.sidebar.success("All records cleared")

# ================= STYLES =================
st.markdown("""
<style>
.card {
    background-color:#111827;
    padding:16px;
    border-radius:12px;
    margin-bottom:15px;
}
</style>
""", unsafe_allow_html=True)

def card(title):
    st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)

# ================= IMAGE PREPROCESS =================
def preprocess_image(img):
    img = np.array(img)

    # If image is already grayscale
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(gray)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


# ================= OCR =================
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\HARINI KAVETI\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"



def extract_text(img):
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6")

# ================= DATE & TIME =================
def extract_date_time(text):
    date = ""
    time = ""

    # ---------- DATE PATTERNS ----------
    date_patterns = [
        r"\b\d{2}/\d{2}/\d{4}\b",      # 26/01/2016
        r"\b\d{2}-\d{2}-\d{4}\b",      # 26-01-2016
        r"\b\d{2}/\d{2}/\d{2}\b",      # 10/18/20
        r"\b\d{2}-\d{2}-\d{2}\b",      # 04-27-19
        r"\b\d{4}-\d{2}-\d{2}\b"       # 2019-04-27
    ]

    for p in date_patterns:
        m = re.search(p, text)
        if m:
            date = m.group()
            break

    # ---------- TIME PATTERN ----------
    m = re.search(r"\b\d{1,2}:\d{2}(:\d{2})?\b", text)
    if m:
        time = m.group()

    return date, time


# ================= FIELD EXTRACTION =================

def extract_merchant(lines):
    ignore_words = [
        "thank", "feedback", "survey", "www", "http",
        "receipt", "chance", "visit", "scan"
    ]

    for line in lines[:10]:  # check only top lines
        l = line.lower()

        # skip junk lines
        if any(word in l for word in ignore_words):
            continue

        # prefer uppercase / brand-like text
        if line.isupper() and len(line) >= 3:
            return line.strip()

    # fallback
    return lines[0] if lines else ""

def generate_invoice_id():
    return "INV-" + uuid.uuid4().hex[:8].upper()

def extract_fields(text):
    data = {
        "invoice_id": generate_invoice_id(),
        "merchant": "",
        "date": "",
        "time": "",
        "subtotal": None,
        "tax": None,
        "total": None,
        "currency": "UNKNOWN",
        "validation_status": "MISMATCH",
        "validation_reason": "",
        "line_items": []
    }

    text_lower = text.lower()
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Merchant
    if lines:
        data["merchant"] = extract_merchant(lines)


    # Date & time
    data["date"], data["time"] = extract_date_time(text)

    # If date looks invalid, clear it
    if not re.match(r"\d", data["date"]):
        data["date"] = ""


    # ---------- Amount extraction ----------
    def extract_amount(patterns):
        for p in patterns:
            m = re.search(p, text_lower)
            if m:
                try:
                    value = re.sub(r"[^\d.]", "", m.group(1))
                    return float(value)


                except:
                    return None
        return None

    data["subtotal"] = extract_amount([
    r"\bsubtotal\b[\s:]*([\d]{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?)"
])


    data["tax"] = extract_amount([r"(tax|gst|vat)[^0-9]*([\d,.]+)"])
    data["total"] = extract_amount([
    r"\btotal\b[\s:]*([\d]{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?)"
])



    # Currency
    if "₹" in text:
        data["currency"] = "INR"
    elif "$" in text:
        data["currency"] = "USD"
    elif "rm" in text_lower:
        data["currency"] = "MYR"




   # ---------- Line items (ROBUST) ----------
    item_lines = []

    # Collect lines BEFORE subtotal
    for l in lines:
        if "subtotal" in l.lower():
            break
        item_lines.append(l)

    ignore_keywords = [
        "total", "tax", "debit", "change", "cash",
        "tend", "ref", "aid", "appr", "code",
        "store", "manager", "phone", "address"
    ]

    for l in item_lines:
        l_clean = re.sub(r"\s+", " ", l.strip())
        l_lower = l_clean.lower()

        if any(k in l_lower for k in ignore_keywords):
            continue

        # Quantity + Item + Price (best case)
        m = re.search(r"^\s*(\d+)\s+(.+?)\s+([\d,.]+\d{2})$", l_clean)
        if m:
            data["line_items"].append({
                "item": m.group(2).strip(),
                "quantity": int(m.group(1)),
                "price": float(m.group(3).replace(",", ""))
            }   )
            continue

        # Item + Price (quantity missing → assume 1)
        m = re.search(r"^\s*([A-Za-z].+?)\s+([\d,.]+\d{2})$", l_clean)
        if m:
            data["line_items"].append({
                "item": m.group(1).strip(),
                "quantity": 1,
                "price": float(m.group(2).replace(",", ""))
            })


    # ---------- Tax rate validation ----------
    if data["subtotal"] is not None and data["tax"] is not None:
        tax_rate = round((data["tax"] / data["subtotal"]) * 100, 2)
        data["tax_rate"] = tax_rate

        if tax_rate < 0 or tax_rate > 30:
            data["validation_status"] = "FAILED"
            data["validation_reason"] = "Invalid tax rate"

    # ---------- Subtotal + Tax = Total ----------
    if data["subtotal"] is not None and data["tax"] is not None and data["total"] is not None:
        calculated_total = round(data["subtotal"] + data["tax"], 2)

        if abs(calculated_total - data["total"]) <= 1:
            data["validation_status"] = "SUCCESS"
            data["validation_reason"] = "Subtotal + Tax matches Total"
        else:
            data["validation_status"] = "FAILED"
            data["validation_reason"] = (
                f"Mismatch: Subtotal({data['subtotal']}) + "
                f"Tax({data['tax']}) ≠ Total({data['total']})"
            )

    elif data["total"] is not None:
        data["validation_status"] = "SUCCESS"
        data["validation_reason"] = "Only total found; subtotal/tax missing"

        # ---------- FALLBACK RULES ----------

    # 1️⃣ If subtotal missing but total present
    if data["subtotal"] is None and data["total"] is not None:
        data["subtotal"] = data["total"]

    # 2️⃣ If total missing but subtotal present
    if data["total"] is None and data["subtotal"] is not None:
        data["total"] = data["subtotal"]

    # 3️⃣ If tax missing, assume 0.0
    if data["tax"] is None:
        data["tax"] = 0.0


    # ---------- HANDLE MISSING SUBTOTAL ----------
    if data["subtotal"] is None and data["total"] is not None:
        data["subtotal"] = data["total"]


    # ---------- HANDLE NET TOTAL ----------
    if data["total"] is None:
        data["total"] = extract_amount([
        r"\bnet\s*total\b[\s:]*([\d]{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?)"
    ])



        # ---------- FORCE TOTAL = SUBTOTAL + TAX ----------
    if data["subtotal"] is not None and data["tax"] is not None:
        calculated_total = round(data["subtotal"] + data["tax"], 2)

        data["total"] = calculated_total
        data["validation_status"] = "SUCCESS"
        data["validation_reason"] = "Total calculated as Subtotal + Tax"
    else:
        data["validation_status"] = "FAILED"
        data["validation_reason"] = "Subtotal or Tax missing, cannot calculate total"


    # ---------- Required fields ----------
    required = ["merchant", "date", "total"]
    missing = [f for f in required if not data[f]]

    if missing:
        data["validation_status"] = "FAILED"
        data["validation_reason"] = f"Missing fields: {', '.join(missing)}"

    return data


# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect("receipts.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS receipts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant TEXT,
            date TEXT,
            time TEXT,
            subtotal REAL,
            tax REAL,
            total REAL,
            currency TEXT,
            validation_status TEXT,
            validation_reason TEXT,
            raw_json TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def is_duplicate(m, d, t):
    conn = sqlite3.connect("receipts.db")
    cur = conn.execute("""
        SELECT 1 FROM receipts
        WHERE merchant=? AND date=? AND total=?
    """, (m, d, t))
    exists = cur.fetchone()
    conn.close()
    return exists is not None


def save_to_db(data):
    conn = sqlite3.connect("receipts.db")
    conn.execute("""
        INSERT INTO receipts VALUES(NULL,?,?,?,?,?,?,?,?,?,?)
    """,(
        data["merchant"],data["date"],data["time"],
        data["subtotal"],data["tax"],data["total"],
        data["currency"],data["validation_status"],
        data["validation_reason"],json.dumps(data)
    ))
    conn.commit()
    conn.close()

# ================= UI =================
st.title("💾 Receipt & Invoice Digitizer")

tab1, tab2, tab3 = st.tabs(["📥 Upload" ,"📈 Spending Insights",
    "🗄️ Saved Receipts"])

# ================= TAB 1 =================
with tab1:
    card("📤 Upload → Extract → Validate → Save")

    file = st.file_uploader("Upload receipt", ["jpg", "png", "pdf"])

    if file:
        imgs = convert_from_bytes(file.read(), 300) if file.type == "application/pdf" else [Image.open(file)]

        col1, col2 = st.columns(2)
        col1.image(imgs[0], caption="Original Receipt", width=300)
        col2.image(preprocess_image(imgs[0]), caption="Processed Image", width=300)

        # Session state
        if "extracted_data" not in st.session_state:
            st.session_state.extracted_data = None
        if "raw_text" not in st.session_state:
            st.session_state.raw_text = ""

        colA, colB = st.columns(2)

        # -------- PROCESS BUTTON --------
        with colA:
            if st.button("🔍 Process Receipt"):
                text = "".join(extract_text(preprocess_image(i)) for i in imgs)
                data = extract_fields(text)

                st.session_state.extracted_data = data
                st.session_state.raw_text = text

                # ✅ Validation message only
                if data["validation_status"] == "SUCCESS":
                    st.success("✅ Validation Successful")
                else:
                    st.error("❌ Validation Failed")

                # -------- EXTRACTED DETAILS --------
                st.subheader("📄 Extracted Details")
                st.write("**Merchant:**", data["merchant"] or "Not detected")
                st.write("**Date:**", data["date"] or "Not detected")
                st.write("**Subtotal:**", data["subtotal"] if data["subtotal"] is not None else "Not detected")
                st.write("**Tax:**", data["tax"] if data["tax"] is not None else "Not detected")
                st.write("**Total:**", data["total"] if data["total"] is not None else "Not detected")
                st.write("**Currency:**", data["currency"])
                st.write("**Invoice ID:**", data["invoice_id"])

                # -------- LINE ITEMS --------
                if data["line_items"]:
                    st.subheader("🧾 Line Items")
                    df_items = pd.DataFrame(data["line_items"])
                    df_items = df_items[["item", "quantity", "price"]]
                    st.dataframe(df_items, use_container_width=True)
                else:
                    st.warning("No line items detected")

                
        # -------- SAVE BUTTON --------
        with colB:
            if st.button("💾 Save to Database"):
                if st.session_state.extracted_data is None:
                    st.warning("Please process the receipt first.")
                else:
                    data = st.session_state.extracted_data

                    if is_duplicate(data["merchant"], data["date"], data["total"]):
                        st.warning("Duplicate receipt. Not saved.")
                    else:
                        save_to_db(data)
                        st.success("Receipt saved successfully.")



# ================= TAB 2 =================
with tab2:
    card("📊 Spending Insights")

    # Load data
    conn = sqlite3.connect("receipts.db")
    df = pd.read_sql(
        "SELECT merchant, total FROM receipts WHERE validation_status='SUCCESS'",
        conn
    )
    conn.close()

    if df.empty:
        st.info("No spending data available.")
        st.stop()

    # Group spending by merchant
    spend_by_merchant = (
        df.groupby("merchant", as_index=False)["total"]
        .sum()
        .sort_values(by="total", ascending=False)
    )

    st.subheader("💰 Spending Summary")

    # -------- BAR CHART (SMALL & CLEAN) --------
    st.markdown("### 📊 Total Expenses per Merchant")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(
            spend_by_merchant.set_index("merchant")["total"],
            use_container_width=True
        )

    # -------- PIE CHART (SMALL) --------
    st.markdown("### 🥧 Spending Distribution by Merchant")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        spend_by_merchant["total"],
        labels=spend_by_merchant["merchant"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)

    # -------- SUMMARY TABLE --------
    st.markdown("### 📄 Spending Table")
    st.dataframe(spend_by_merchant, use_container_width=True)


# ================= TAB 3 =================
with tab3:
    card("🗄️ Saved Receipts")

    st.subheader("📦 Table: receipts")

    conn = sqlite3.connect("receipts.db")
    df = pd.read_sql("SELECT * FROM receipts ORDER BY id DESC", conn)
    conn.close()

    if df.empty:
        st.info("No receipts found in database.")
        st.stop()

    # ================= SEARCH BY RECEIPT ID =================
    st.markdown("### 🔍 Search by Receipt ID")

    search_id = st.text_input("Enter Receipt ID (Invoice ID)")

    if search_id:
        search_df = df[df["raw_json"].str.contains(search_id, case=False, na=False)]

        if search_df.empty:
            st.warning("No records found for this Receipt ID.")
        else:
            st.markdown("#### 📄 Search Result")
            st.dataframe(search_df, use_container_width=True)

    # ================= FILTER BY DATE =================
    st.markdown("### 📅 Filter by Date")

    from datetime import date

    col1, col2 = st.columns(2)

    from_date = col1.date_input(
        "From Date",
        value=None
    )

    to_date = col2.date_input(
        "To Date",
        value=None
    )



    date_df = df.copy()

    if from_date:
        date_df = date_df[date_df["date"] >= from_date]
    if to_date:
        date_df = date_df[date_df["date"] <= to_date]

    if from_date or to_date:
        if date_df.empty:
            st.warning("No records found for selected date range.")
        else:
            st.markdown("#### 📄 Date Filter Result")
            st.dataframe(date_df, use_container_width=True)

    # ================= FULL TABLE =================
    st.markdown("### 📊 Stored Receipts Data (All Records)")
    st.dataframe(df, use_container_width=True)

    # ================= DELETE RECORD =================
st.markdown("### 🗑️ Delete Receipt by ID")

col1, col2 = st.columns([2, 1])

with col1:
    delete_id = st.text_input("Enter Record ID to delete")

with col2:
    if st.button("❌ Delete"):
        if not delete_id:
            st.warning("Please enter a Record ID.")
        else:
            conn = sqlite3.connect("receipts.db")
            cur = conn.execute(
                "SELECT 1 FROM receipts WHERE id = ?",
                (delete_id,)
            )
            exists = cur.fetchone()

            if not exists:
                st.error("Record ID not found.")
            else:
                conn.execute(
                    "DELETE FROM receipts WHERE id = ?",
                    (delete_id,)
                )
                conn.commit()
                st.success(f"Record ID {delete_id} deleted successfully.")

            conn.close()

