# app.py
import streamlit as st
from openai import OpenAI
import sqlite3
import pandas as pd
import json
import re
import io
import os
from PIL import Image
import base64
import traceback

# ---------------- CONFIG ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("⚠️ Please set OPENAI_API_KEY in Streamlit secrets or environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DB_PATH = "invoices.db"

# ---------------- DATABASE SETUP ----------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS invoice_line_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    marketplace_name TEXT,
    invoice_type TEXT,
    invoice_date TEXT,
    place_of_supply TEXT,
    gstin TEXT,
    service_description TEXT,
    net_taxable_value REAL,
    total_tax_rate TEXT,
    total_amount REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
conn.commit()

# ---------------- PROMPT TEMPLATE ----------------
PROMPT = """
You are given an image of an invoice or credit note from Amazon or Flipkart.
Extract and return all the line items as a JSON array.
Each object should represent one service row from the invoice or credit note.

Return STRICT JSON ONLY (no markdown, no explanation).
If any field is missing, set it to null.

JSON keys for each object:
[
  {
    "Marketplace Name": string | null,
    "Types of Invoice": "Tax Invoice" | "Credit Note" | "Commercial Credit Note" | string,
    "Date of Invoice/Credit Note": string | null,
    "Place of Supply": string | null,
    "GSTIN": string | null,
    "Service Description": string,
    "Net Taxable Value": number | string | null,
    "Total Tax Rate": string | null,
    "Total Amount": number | string | null
  }
]

Guidelines:
- Identify invoice type from document heading ("Tax Invoice", "Credit Note", etc.)
- Include one object per service row (e.g. Pick & Pack Fee, Shipping Fee)
- Use percentage format for tax rate (e.g. "18%" or "9%+9%")
- Include total values as shown in invoice currency
- Return valid JSON only, nothing else.
"""

# ---------------- HELPERS ----------------
def extract_json(text):
    """Extract JSON array from LLM output"""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text, re.S)
        if match:
            try:
                cleaned = re.sub(r",\s*([}\]])", r"\1", match.group(0))
                return json.loads(cleaned)
            except:
                pass
    return None


def sanitize_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    s = re.sub(r"[^\d.\-]", "", str(value))
    if not s:
        return None
    try:
        return float(s)
    except:
        return None


def insert_rows(rows):
    for r in rows:
        try:
            cur.execute("""
                INSERT INTO invoice_line_items
                (marketplace_name, invoice_type, invoice_date, place_of_supply, gstin,
                 service_description, net_taxable_value, total_tax_rate, total_amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r.get("Marketplace Name"),
                r.get("Types of Invoice"),
                r.get("Date of Invoice/Credit Note"),
                r.get("Place of Supply"),
                r.get("GSTIN"),
                r.get("Service Description"),
                sanitize_number(r.get("Net Taxable Value")),
                r.get("Total Tax Rate"),
                sanitize_number(r.get("Total Amount")),
            ))
        except Exception as e:
            st.error(f"DB insert failed: {e}")
    conn.commit()


def delete_row(row_id):
    try:
        cur.execute("DELETE FROM invoice_line_items WHERE id = ?", (row_id,))
        conn.commit()
    except Exception as e:
        st.error(f"Failed to delete row: {e}")


def fetch_all_rows(limit=500):
    return pd.read_sql_query(
        "SELECT * FROM invoice_line_items ORDER BY created_at DESC LIMIT ?",
        conn,
        params=(limit,)
    )


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to base64 string for OpenAI API"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def call_openai_vision(image: Image.Image):
    """Send image to OpenAI Vision model"""
    img_b64 = image_to_base64(image)
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": PROMPT.strip()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]}
            ],
            temperature=0,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": PROMPT.strip()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]}
            ],
            temperature=0,
            max_tokens=1500
        )
        return response.choices[0].message.content


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Marketplace Invoice Parser", layout="wide")
st.title("🧾 Marketplace Invoice Parser (Amazon & Flipkart)")

st.markdown("""
Upload **one invoice image (JPG/PNG)**.  
The app sends it to **OpenAI Vision (GPT-4o)** which reads and extracts structured data automatically — no OCR engine required.
""")

uploaded_file = st.file_uploader("Upload Invoice Image", type=["jpg", "jpeg", "png"])
parse_button = st.button("Parse & Save Data")

if parse_button:
    if not uploaded_file:
        st.warning("Please upload an image first.")
    elif not OPENAI_API_KEY:
        st.error("Set your OPENAI_API_KEY in Streamlit Secrets or environment.")
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            with st.spinner("🔍 Sending image to OpenAI Vision..."):
                llm_output = call_openai_vision(image)

            parsed = extract_json(llm_output)
            if not parsed:
                st.error("❌ Could not parse valid JSON. Try again or check the image.")
            else:
                insert_rows(parsed)
                st.success(f"✅ Parsed and saved {len(parsed)} line items successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
            st.error(traceback.format_exc())

st.markdown("---")
st.subheader("📊 Stored Invoice Line Items")

df = fetch_all_rows()
if df.empty:
    st.info("No records yet. Upload an invoice to begin.")
else:
    # Create a delete button for each row
    for _, row in df.iterrows():
        cols = st.columns([1, 1.2, 1.2, 1.2, 1.2, 2, 1.2, 1.2, 0.3])
        cols[0].write(f"**{row['marketplace_name'] or ''}**")
        cols[1].write(row['invoice_type'])
        cols[2].write(row['invoice_date'])
        cols[3].write(row['place_of_supply'])
        cols[4].write(row['gstin'])
        cols[5].write(row['service_description'])
        cols[6].write(row['net_taxable_value'])
        cols[7].write(row['total_amount'])
        if cols[8].button("🗑️", key=f"delete_{row['id']}"):
            delete_row(row["id"])
            st.rerun()
