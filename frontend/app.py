# frontend/app.py
import os
import requests
import pandas as pd
import streamlit as st

from llm_client import (
    extract_structured_with_taxonomy,
    extract_object,
    extract_three_terms,
    summarize_products,
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="LLM + Cosine Search", page_icon="🛍️", layout="centered")
st.title("🛍️ LLM-powered Product Finder")
st.caption("We parse your intent, derive an object and three terms, filter by taxonomy/type, then rank by cosine + business weights.")

# ---- Discover taxonomy from backend
tax = {"category_name_1": [], "category_name_2": [], "category_name_3": [], "brand": []}
try:
    tax = requests.get(f"{BACKEND_URL}/taxonomy", timeout=10).json()
except Exception:
    pass

cat1_opts = tax.get("category_name_1", [])
cat2_opts = tax.get("category_name_2", [])
cat3_opts = tax.get("category_name_3", [])
brand_opts = tax.get("brand", [])

# ---- Sidebar: ONLY the two business weights
with st.sidebar:
    st.subheader("Business Weights")
    profitability_w = st.slider("Profitability weighting", 0.0, 1.0, 0.0, 0.05)
    return_rate_w  = st.slider("Return-rate weighting",   0.0, 1.0, 0.0, 0.05)
    ALPHA_DEFAULT = st.slider("Cosine vs Business (alpha)", 0.0, 1.0, 0.70, 0.05)

ALPHA_DEFAULT = 0.70

# ---- Main form
with st.form(key="query_form"):
    user_text = st.text_input(
        "What are you looking for?",
        placeholder="e.g., dash cam under $150, or nike red shoes size 10"
    )
    top_k = st.number_input("How many results?", min_value=1, max_value=200, value=5, step=1)
    submitted = st.form_submit_button("Search")

if submitted and user_text.strip():
    # 1) Category (broad)
    with st.spinner("Classifying your category..."):
        intent = extract_structured_with_taxonomy(user_text, cat1_opts, cat2_opts, cat3_opts, brand_opts)

    # 2) Object + three descriptive terms (using your exact prompts)
    with st.spinner("Extracting object & terms..."):
        obj = extract_object(user_text)
        pos_terms = extract_three_terms(obj)

    st.write("**Parsed intent:**", {
        **intent,
        "object": obj,
        "pos_terms": pos_terms
    })

    # 3) Build payload for backend
    keywords = " ".join(pos_terms) if pos_terms else user_text
    payload = {
        "query": keywords,
        "pos_terms": pos_terms,                     # <-- NEW: three positional terms
        "top_k": int(top_k),
        "alpha": float(ALPHA_DEFAULT),
        "brand": intent.get("brand") or None,
        "color": intent.get("color") or None,
        "object": obj or None,                      # used by backend candidate filtering
        "category_any": intent.get("category_name_1") or None,
        "category_name_1": None,
        "category_name_2": None,
        "category_name_3": None,
    }

    # Business weights
    biz_weights = {
        "profitability": float(profitability_w),
        "return_rate": float(return_rate_w),
    }
    if any(v > 0 for v in biz_weights.values()):
        payload["biz_weights"] = biz_weights

    # 4) Query backend
    with st.spinner("Searching catalog..."):
        items = []
        try:
            r = requests.post(f"{BACKEND_URL}/search", json=payload, timeout=30)
            r.raise_for_status()
            items = r.json().get("items", [])
        except Exception as e:
            st.error(f"Backend error: {e}")

    # 5) Render
    if items:
        st.caption(
            f"Applied → brand: {payload.get('brand') or '—'}, color: {payload.get('color') or '—'}, "
            f"object: {payload.get('object') or '—'}, category_any: {payload.get('category_any') or '—'}, "
            f"terms: {', '.join(payload.get('pos_terms') or []) or '—'}"
        )
        df = pd.DataFrame(items)
        cols = [c for c in ["score","similarity","business_score","product_id","name","brand","current_price","product_url"] if c in df.columns]
        st.dataframe(df[cols] if cols else df, use_container_width=True)

        with st.spinner("Writing a readable summary..."):
            summary = summarize_products(user_text, keywords, items)
        st.markdown("---")
        st.markdown("### Summary")
        st.write(summary)
    else:
        st.info("No results found. Try different wording or relax filters.")

else:
    st.write("Enter a product request above and press **Search**.")
