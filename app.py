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

# ---------------- PROMPT ----------------
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
"""

# ---------------- HELPERS ----------------
def extract_json(text):
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
    conn.commit()


def delete_row(row_id):
    cur.execute("DELETE FROM invoice_line_items WHERE id = ?", (row_id,))
    conn.commit()


def fetch_all_rows():
    return pd.read_sql_query(
        "SELECT id, marketplace_name, invoice_type, invoice_date, place_of_supply, gstin, "
        "service_description, net_taxable_value, total_tax_rate, total_amount "
        "FROM invoice_line_items ORDER BY created_at DESC",
        conn
    )


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def call_openai_vision(image: Image.Image):
    img_b64 = image_to_base64(image)
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


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Marketplace Invoice Parser", layout="wide")
st.title("🧾 Marketplace Invoice Parser (Amazon & Flipkart)")

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
            with st.spinner("🔍 Processing image..."):
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
    st.download_button(
        label="⬇️ Download Table as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="invoice_data.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Delete button column
    df["Delete"] = df["id"].apply(lambda i: f"🗑️ Delete {i}")

    # Display bordered table with headers
    st.dataframe(
        df.drop(columns=["id"]),
        use_container_width=True,
        hide_index=True
    )

    # Handle row deletion
    for i, row in df.iterrows():
        delete_key = f"delete_{row['id']}"
        if st.button("🗑️", key=delete_key):
            delete_row(row["id"])
            st.rerun()
