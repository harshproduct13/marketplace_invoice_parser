# app.py
import streamlit as st
from PIL import Image
import pytesseract
import sqlite3
import json
import re
import os
import pandas as pd
import traceback
import openai

# ---------- CONFIG ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("⚠️ Please set the environment variable OPENAI_API_KEY before running the app.")
openai.api_key = OPENAI_API_KEY

# Use fast model by default, fallback to gpt-4o if not found
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

DB_PATH = "invoices.db"
MAX_OCR_CHARS = 40000  # Limit OCR text for token efficiency

# ---------- DATABASE SETUP ----------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute(
    """
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
    raw_text TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""
)
conn.commit()

# ---------- PROMPT TEMPLATE ----------
PROMPT_TEMPLATE = r"""
You will be given the full OCR-extracted text from a marketplace invoice or credit note (Amazon or Flipkart).
Your job is to RETURN STRICT JSON ONLY: an ARRAY of objects where each object represents ONE service line from the invoice
(e.g., Pick & Pack Fee, Commission Fee, Shipping Fee).

Each object must have these exact keys:
[
  {
    "Marketplace Name": string or null,
    "Types of Invoice": "Tax Invoice" | "Credit Note" | "Commercial Credit Note" | string,
    "Date of Invoice/Credit Note": string or null,
    "Place of Supply": string or null,
    "GSTIN": string or null,
    "Service Description": string,
    "Net Taxable Value": number or string (numeric) or null,
    "Total Tax Rate": string like "18%" or "9%+9%" or null,
    "Total Amount": number or string (numeric) or null
  }
]

Rules:
1. Return JSON only and nothing else. If you can't find a field, return null for it.
2. If the document explicitly shows the words "Commercial Credit Note" and there is no tax column in the breakdown,
   set "Types of Invoice" to "Commercial Credit Note", set "Net Taxable Value" and "Total Tax Rate" to null,
   but still return "Total Amount" and "Service Description" rows.
3. If the invoice is a Tax Invoice / Commission Invoice, fill tax rate (e.g. "18%" or "9%+9%") and numeric fields where possible.
4. Date must be the invoice/credit note date shown on the document — return as-is if you cannot normalize.
5. Split every service row into a separate object; reuse the invoice-level fields (Marketplace Name, Date, Place of Supply, GSTIN) for each service row.

Now parse the following text (the OCR-extracted text is delimited by triple quotes). Be accurate and return valid JSON array.

\"\"\"<<OCR_TEXT>>\"\"\"
"""

# ---------- HELPERS ----------
def extract_json_from_text(text):
    """Try to extract valid JSON array from LLM output."""
    try:
        return json.loads(text)
    except:
        pass
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
    if s == "":
        return None
    try:
        return float(s)
    except:
        return None


def call_openai_parse(ocr_text):
    """Send OCR text to OpenAI and return model output."""
    prompt = PROMPT_TEMPLATE.replace("<<OCR_TEXT>>", ocr_text)
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a JSON extractor for invoice data."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        return response["choices"][0]["message"]["content"]
    except Exception:
        # fallback to gpt-4o
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a JSON extractor for invoice data."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        return response["choices"][0]["message"]["content"]


def insert_rows_to_db(rows, raw_text):
    """Insert parsed rows into SQLite DB."""
    cur = conn.cursor()
    for r in rows:
        try:
            cur.execute(
                """
                INSERT INTO invoice_line_items
                (marketplace_name, invoice_type, invoice_date, place_of_supply, gstin, service_description,
                net_taxable_value, total_tax_rate, total_amount, raw_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.get("Marketplace Name"),
                    r.get("Types of Invoice"),
                    r.get("Date of Invoice/Credit Note"),
                    r.get("Place of Supply"),
                    r.get("GSTIN"),
                    r.get("Service Description"),
                    sanitize_number(r.get("Net Taxable Value")),
                    r.get("Total Tax Rate"),
                    sanitize_number(r.get("Total Amount")),
                    raw_text[:20000],
                ),
            )
            conn.commit()
        except Exception as e:
            st.error(f"❌ Failed to insert row: {e}")


def fetch_all_rows(limit=1000):
    df = pd.read_sql_query(
        "SELECT * FROM invoice_line_items ORDER BY created_at DESC LIMIT ?", conn, params=(limit,)
    )
    return df


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Marketplace Invoice Parser", layout="wide")
st.title("🧾 Marketplace Invoice Parser (Amazon & Flipkart)")

st.markdown(
    """
Upload **one invoice image (JPG/PNG)** at a time.  
The app will:
1. Run OCR using Tesseract  
2. Send extracted text to OpenAI  
3. Parse all service line-items into structured data  
4. Store and display results below  
"""
)

uploaded_file = st.file_uploader("Upload Invoice Image", type=["jpg", "jpeg", "png"])
parse_button = st.button("Parse & Save Data")

if parse_button:
    if not uploaded_file:
        st.warning("Please upload an image first.")
    elif not OPENAI_API_KEY:
        st.error("Set the environment variable OPENAI_API_KEY.")
    else:
        with st.spinner("🧠 Running OCR and sending to OpenAI..."):
            try:
                image = Image.open(uploaded_file).convert("RGB")
                ocr_text = pytesseract.image_to_string(image)
                ocr_text = ocr_text[:MAX_OCR_CHARS]
                if not ocr_text.strip():
                    st.warning("OCR returned empty text. Try a clearer image.")
                    st.stop()

                st.subheader("Extracted OCR Text")
                st.text_area("OCR Output", ocr_text, height=250)

                llm_output = call_openai_parse(ocr_text)
                st.subheader("LLM Raw Output")
                st.code(llm_output[:2000])

                parsed = extract_json_from_text(llm_output)
                if not parsed:
                    st.error("Could not parse valid JSON. Check LLM output above.")
                else:
                    insert_rows_to_db(parsed, ocr_text)
                    st.success(f"✅ Parsed and saved {len(parsed)} rows successfully.")
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.error(traceback.format_exc())

st.markdown("---")
st.subheader("📊 Stored Invoice Line Items")

df = fetch_all_rows()
if df.empty:
    st.info("No records yet. Upload an invoice to get started.")
else:
    st.dataframe(df, use_container_width=True)
