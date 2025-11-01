# app.py
import streamlit as st
from PIL import Image
import pytesseract
import sqlite3
import json
import re
import os
from datetime import datetime
import pandas as pd
import io
import tempfile
import traceback

# OpenAI import (make sure you have openai>=1.0.0 installed)
import openai

# ---------- CONFIG ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Set OPENAI_API_KEY environment variable before using the app.")
openai.api_key = OPENAI_API_KEY

# Default model (you can change to "gpt-4o" if you want higher accuracy)
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

DB_PATH = "invoices.db"
MAX_OCR_CHARS = 40000  # trim OCR if extremely long

# ---------- DB SETUP ----------
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
You will be given the full OCR-extracted text from a marketplace invoice or credit note (Amazon or Flipkart). Your job is to RETURN STRICT JSON ONLY: an ARRAY of objects where each object represents ONE service line from the invoice (e.g., Pick & Pack Fee, Commission Fee, Shipping Fee).

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
1. Return JSON only and nothing else. No explanations. If you can't find a field, return null for it.
2. If the document explicitly shows the words "Commercial Credit Note" and there is no tax column in the breakdown, set "Types of Invoice" to "Commercial Credit Note", set "Net Taxable Value" and "Total Tax Rate" to null, but still return "Total Amount" and "Service Description" rows.
3. If the invoice is a Tax Invoice / Commission Invoice, fill tax rate (e.g. "18%" or "9%+9%") and numeric fields where possible.
4. Date must be the invoice/credit note date shown on the document — return as-is if you cannot normalize.
5. Split every service row into a separate object; reuse the invoice-level fields (Marketplace Name, Date, Place of Supply, GSTIN) for each service row.

Now parse the following text (the OCR-extracted text is delimited by triple quotes). Be accurate and return valid JSON array.

""" 
<<OCR_TEXT>>
"""
"""

# ---------- HELPERS ----------
def extract_json_from_text(text):
    """
    Try to extract a JSON array from the LLM text, handling code fences or extra commentary.
    """
    # direct parse attempt
    try:
        return json.loads(text)
    except Exception:
        pass

    # remove common prefixes/suffixes and try again
    # 1) extract first JSON array if present
    arr_match = re.search(r"(\[\s*\{[\s\S]*?\}\s*\])", text)
    if arr_match:
        try:
            return json.loads(arr_match.group(1))
        except Exception:
            pass

    # 2) remove markdown code fences
    text2 = re.sub(r"```(?:json)?", "", text)
    text2 = text2.replace("```", "")
    try:
        return json.loads(text2)
    except Exception:
        pass

    # 3) try to find a JSON-like structure and fix trailing commas
    candidate = re.search(r"\[.*\]", text2, flags=re.S)
    if candidate:
        cand = candidate.group(0)
        # remove trailing commas before } or ]
        cand = re.sub(r",\s*([}\]])", r"\1", cand)
        try:
            return json.loads(cand)
        except Exception:
            pass

    return None


def sanitize_number(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip()
    if s == "":
        return None
    # remove common currency symbols and commas
    s2 = re.sub(r"[^\d.\-]", "", s)
    if s2 == "":
        return None
    try:
        if "." in s2:
            return float(s2)
        else:
            return int(s2)
    except:
        try:
            return float(s2)
        except:
            return None


def call_openai_parse(ocr_text):
    """
    Send the OCR text with prompt to OpenAI chat completion and return text response.
    """
    prompt = PROMPT_TEMPLATE.replace("<<OCR_TEXT>>", ocr_text)
    # Use ChatCompletion (OpenAI python package compatibility)
    # We'll call the Chat Completions endpoint via openai.ChatCompletion.create if available
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict JSON extractor for marketplace invoices."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        text = resp["choices"][0]["message"]["content"]
        return text
    except Exception as e:
        # Fallback to "responses" or openai.Completion usage if API client differs
        # Try the newer client call (if installed)
        try:
            resp = openai.Completion.create(
                engine=OPENAI_MODEL,
                prompt=prompt,
                max_tokens=1500,
                temperature=0.0,
            )
            return resp["choices"][0]["text"]
        except Exception as e2:
            raise e  # re-raise original


def insert_rows_to_db(rows, raw_text):
    inserted = []
    cur = conn.cursor()
    for r in rows:
        try:
            marketplace_name = r.get("Marketplace Name") or r.get("marketplace_name")
            invoice_type = r.get("Types of Invoice") or r.get("Types of invoice") or r.get("invoice_type")
            invoice_date = r.get("Date of Invoice/Credit Note") or r.get("Date") or r.get("invoice_date")
            place_of_supply = r.get("Place of Supply") or r.get("place_of_supply")
            gstin = r.get("GSTIN") or r.get("gstin")
            service_description = r.get("Service Description") or r.get("service_description") or ""
            net_taxable = sanitize_number(r.get("Net Taxable Value") or r.get("NetTaxable") or r.get("net_taxable_value"))
            total_tax_rate = r.get("Total Tax Rate") or r.get("total_tax_rate")
            total_amount = sanitize_number(r.get("Total Amount") or r.get("TotalAmount") or r.get("total_amount"))

            cur.execute(
                """
                INSERT INTO invoice_line_items
                (marketplace_name, invoice_type, invoice_date, place_of_supply, gstin, service_description, net_taxable_value, total_tax_rate, total_amount, raw_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    marketplace_name,
                    invoice_type,
                    invoice_date,
                    place_of_supply,
                    gstin,
                    service_description,
                    net_taxable,
                    total_tax_rate,
                    total_amount,
                    raw_text[:20000],
                ),
            )
            conn.commit()
            lastid = cur.lastrowid
            inserted.append(lastid)
        except Exception:
            st.error("Failed to insert one row to DB: " + traceback.format_exc())
    return inserted


def fetch_all_rows(limit=1000):
    df = pd.read_sql_query(
        "SELECT id, marketplace_name, invoice_type, invoice_date, place_of_supply, gstin, service_description, net_taxable_value, total_tax_rate, total_amount, created_at FROM invoice_line_items ORDER BY created_at DESC LIMIT ?",
        conn,
        params=(limit,),
    )
    return df


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Marketplace Invoice Parser (Images)", layout="wide")
st.title("Marketplace Invoice Parser — (Amazon & Flipkart invoices)")

st.markdown(
    """
Upload a single invoice image (jpg/png). The app runs OCR on the image and asks an LLM to extract structured rows.
Each service description in the invoice will become one row in the table.
"""
)

col1, col2 = st.columns([2, 3])

with col1:
    uploaded_file = st.file_uploader("Upload invoice image (one at a time)", type=["png", "jpg", "jpeg"])
    model_choice = st.selectbox("LLM Model (change if you set env OPENAI_MODEL)", options=[OPENAI_MODEL, "gpt-4o"], index=0)
    parse_button = st.button("Parse image & Save rows")

with col2:
    st.info("Make sure you set OPENAI_API_KEY env var. OCR uses Tesseract (system binary).")

if uploaded_file:
    # show preview
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded image preview", use_column_width=True)
    except Exception as e:
        st.error("Cannot open image: " + str(e))
        st.stop()

if parse_button:
    if not uploaded_file:
        st.warning("Please upload an image first.")
    elif not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY environment variable not set.")
    else:
        with st.spinner("Running OCR and parsing with LLM..."):
            try:
                # Save temp image for pytesseract
                img_bytes = uploaded_file.read()
                tmp = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                # OCR: you must have tesseract installed on your system
                raw_text = pytesseract.image_to_string(tmp)
                if not raw_text or raw_text.strip() == "":
                    st.warning("OCR returned empty text. Try a clearer image or crop to invoice area.")
                else:
                    # trim to MAX_OCR_CHARS to reduce token usage
                    ocr_text = raw_text[:MAX_OCR_CHARS]

                    st.subheader("OCR Extracted Text (truncated)")
                    st.text_area("OCR text", ocr_text, height=250)

                    # call OpenAI
                    try:
                        # set model if changed
                        OPENAI_MODEL = model_choice
                        llm_output = call_openai_parse(ocr_text)
                    except Exception as e:
                        st.error("OpenAI call failed: " + str(e))
                        st.stop()

                    st.subheader("LLM raw output (preview)")
                    st.code(llm_output[:4000])

                    # extract JSON
                    parsed = extract_json_from_text(llm_output)
                    if parsed is None:
                        st.error("Could not parse JSON from LLM response. See LLM output above.")
                    else:
                        # Validate each object contains Service Description
                        final_rows = []
                        for obj in parsed:
                            # ensure keys exist
                            service_desc = obj.get("Service Description") or obj.get("service_description") or ""
                            if not service_desc:
                                # skip rows with empty service description
                                continue
                            final_rows.append(obj)

                        if not final_rows:
                            st.warning("No valid line items found in LLM JSON output.")
                        else:
                            # insert to db
                            inserted_ids = insert_rows_to_db(final_rows, ocr_text)
                            st.success(f"Inserted {len(inserted_ids)} rows to DB.")
                            st.experimental_rerun()
            except Exception as e:
                st.error("Error during processing: " + str(e))
                st.error(traceback.format_exc())

# display stored rows
st.markdown("---")
st.subheader("Stored Line Items (most recent first)")
df = fetch_all_rows(1000)
if df.empty:
    st.info("No rows yet — upload an image and parse.")
else:
    # prettify columns
    df_display = df.copy()
    df_display["net_taxable_value"] = df_display["net_taxable_value"].map(lambda x: "" if x is None else x)
    df_display["total_amount"] = df_display["total_amount"].map(lambda x: "" if x is None else x)
    st.dataframe(df_display, use_container_width=True)

st.markdown(
    """
**Notes**
- This app uses pytesseract for OCR. You must install the Tesseract binary on your machine (instructions below).
- The LLM prompt requests strict JSON, but LLMs sometimes return extra text — the app tries to extract JSON robustly.
- If you want better OCR or accuracy, you can pre-crop the invoice image or use higher-quality images.
"""
)

st.markdown("### Troubleshooting & tips")
st.markdown(
    """
- If OCR returns garbage, install language packs for tesseract or increase image dpi.
- To change to a different OpenAI model, set the `OPENAI_MODEL` environment variable or select in the UI.
- Save your DB or back it up: `invoices.db`.
"""
)
