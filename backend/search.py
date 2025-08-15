# backend/search.py
from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========= Cleaning / normalization =========
EXCEL_ERR = re.compile(r"^\s*#(?:REF|NAME|VALUE|NULL|N/?A|DIV/0!?|NUM|CALC)!?\s*$", re.I)

def _clean_token_text(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if EXCEL_ERR.match(s) else s

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\-\/\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _plural_to_singular(q: str) -> str:
    toks, out = (q or "").split(), []
    for t in toks:
        if len(t) > 3 and t.endswith("es") and not t.endswith("ses"):
            out.append(t[:-2])
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            out.append(t[:-1])
        else:
            out.append(t)
    return " ".join(out)

def _minmax(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    lo, hi = x.min(), x.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - lo) / (hi - lo)

def _norm_eq(series: pd.Series, val: Optional[str]) -> np.ndarray:
    if not val:
        return np.ones(len(series), dtype=bool)
    v = val.strip().lower()
    return series.astype(str).str.strip().str.lower().eq(v).to_numpy()

# ========= Color aliases (optional prefiltering) =========
COLOR_ALIASES: Dict[str, set[str]] = {
    "red": {"red", "crimson", "scarlet", "maroon", "burgundy", "ruby"},
    "blue": {"blue", "navy", "royal", "cobalt", "azure"},
    "black": {"black"},
    "white": {"white"},
    "green": {"green", "emerald", "lime", "olive"},
    "pink": {"pink", "rose", "magenta", "fuchsia"},
    "purple": {"purple", "violet", "lilac"},
    "yellow": {"yellow", "golden"},
    "orange": {"orange", "tangerine"},
    "brown": {"brown", "chocolate", "tan"},
    "gray": {"gray", "grey", "charcoal"},
    "silver": {"silver"},
    "gold": {"gold", "golden"},
}

def _contains_any(text: str, vocab: set[str]) -> bool:
    if not text:
        return False
    words = set(re.split(r"[^\w]+", text.lower()))
    return bool(words & vocab)

# ========= Core class =========
class CosineSearch:
    """
    TF-IDF search (name + category text), centroid query made from 4 components
    (query, category_name, object, joined terms), and business-score blending.

    Final score = alpha * cosine + (1 - alpha) * business_score
    """

    # Only the two business features you care about:
    # - profitability: prefer higher margin
    # - return_rate: prefer lower return rate (we invert it)
    DEFAULT_FEATURE_MAP = {
        "profitability": ("_derived_margin", None),
        "return_rate": ("return_rate", "inverse"),
    }

    def __init__(self, csv_path: str, *, name_col: str = "name"):
        self.name_col = name_col
        self.df = pd.read_csv(csv_path)

        # normalize product_id if present
        if "product_id" in self.df.columns:
            self.df["product_id"] = (
                self.df["product_id"].astype(str).str.replace(r"[,\s]", "", regex=True).str.strip()
            )

        # ---------- Merge return rates if available ----------
        rr_path = os.getenv("RETURN_RATES_PATH", "/data/return_rates.pkl")
        if os.path.exists(rr_path):
            try:
                rr = pd.read_pickle(rr_path)
                if "product_id" not in rr.columns:
                    rr = rr.rename_axis("product_id").reset_index()
                rr["product_id"] = (
                    rr["product_id"].astype(str).str.replace(r"[,\s]", "", regex=True).str.strip()
                )
                keep = [c for c in ["product_id", "return_rate"] if c in rr.columns]
                if "product_id" in keep and "return_rate" in keep:
                    self.df = self.df.merge(rr[keep], on="product_id", how="left")
            except Exception:
                # continue without return_rate if load fails
                pass

        # Clean text fields we use
        for col in [
            self.name_col, "brand",
            "category_name_1", "category_name_2", "category_name_3", "category_name_4",
            "request_path_1", "request_path_2", "request_path_3", "request_path_4",
        ]:
            if col in self.df.columns:
                self.df[col] = self.df[col].map(_clean_token_text)

        # Derived profitability = (price - cost) / price if possible.
        if "current_price" in self.df.columns and "current_cost" in self.df.columns:
            pr = pd.to_numeric(self.df["current_price"], errors="coerce")
            ct = pd.to_numeric(self.df["current_cost"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                self.df["_derived_margin"] = np.where(pr > 0, (pr - ct) / pr, np.nan)

        # fallback to current_margin if derived is all NA
        if "_derived_margin" in self.df.columns and "current_margin" in self.df.columns:
            if pd.isna(self.df["_derived_margin"]).all():
                self.df["_derived_margin"] = pd.to_numeric(self.df["current_margin"], errors="coerce")

        # === Build TF-IDF blocks: NAME and CATEGORY BLOB ===
        name_series = self.df[self.name_col].fillna("").astype(str).map(_norm_text)
        self.df["search_name"] = (name_series + " ").str.strip()

        c1 = self.df.get("category_name_1", pd.Series("", index=self.df.index)).fillna("").astype(str).map(_norm_text)
        c2 = self.df.get("category_name_2", pd.Series("", index=self.df.index)).fillna("").astype(str).map(_norm_text)
        c3 = self.df.get("category_name_3", pd.Series("", index=self.df.index)).fillna("").astype(str).map(_norm_text)
        c4 = self.df.get("category_name_4", pd.Series("", index=self.df.index)).fillna("").astype(str).map(_norm_text)
        rp1 = self.df.get("request_path_1", pd.Series("", index=self.df.index)).fillna("").astype(str).map(_norm_text)
        rp2 = self.df.get("request_path_2", pd.Series("", index=self.df.index)).fillna("").astype(str).map(_norm_text)
        rp3 = self.df.get("request_path_3", pd.Series("", index=self.df.index)).fillna("").astype(str).map(_norm_text)
        rp4 = self.df.get("request_path_4", pd.Series("", index=self.df.index)).fillna("").astype(str).map(_norm_text)
        catblob = (c1 + " " + c2 + " " + c3 + " " + c4 + " " + rp1 + " " + rp2 + " " + rp3 + " " + rp4).str.strip()
        self.df["search_cat"] = catblob

        # Vectorizers
        self.v_name = TfidfVectorizer(
            lowercase=True, stop_words="english", strip_accents="unicode",
            ngram_range=(1, 3), sublinear_tf=True, smooth_idf=True,
            min_df=1, max_df=0.95
        )
        self.v_cat = TfidfVectorizer(
            lowercase=True, stop_words="english", strip_accents="unicode",
            ngram_range=(1, 3), sublinear_tf=True, smooth_idf=True,
            min_df=1, max_df=0.95
        )

        N = len(self.df)

        def _fit_or_empty(vec, series):
            try:
                return vec.fit_transform(series)
            except Exception:
                # empty vocab → return (N x 0) block to keep pipeline alive
                return csr_matrix((N, 0))

        Xn = _fit_or_empty(self.v_name, self.df["search_name"])
        Xc = _fit_or_empty(self.v_cat,  self.df["search_cat"])

        # weights: name 5x, categories 2x (tune if you like)
        self.X = hstack([5 * Xn, 2 * Xc]).tocsr()

        # Business features
        self.biz_feature_names, self.biz_matrix = self._build_business_matrix()

    # ---------- Business features ----------
    def _build_business_matrix(self) -> Tuple[List[str], np.ndarray]:
        cols, mats = [], []
        for feat, (col, mode) in self.DEFAULT_FEATURE_MAP.items():
            if col not in self.df.columns:
                continue
            s = pd.to_numeric(self.df[col], errors="coerce")
            s_filled = s.fillna(s.median())
            z = _minmax(s_filled)
            if mode == "inverse":
                z = 1.0 - z
            cols.append(feat)
            mats.append(z.values.reshape(-1, 1))
        if not mats:
            return [], np.zeros((len(self.df), 0))
        M = np.hstack(mats)
        return cols, M

    # ---------- Encoders (project any text into the same space) ----------
    def _encode_text(self, q: str) -> csr_matrix:
        q = _plural_to_singular(_norm_text(q or ""))
        qn = self.v_name.transform([q])
        qc = self.v_cat.transform([q])
        return hstack([5 * qn, 2 * qc]).tocsr()

    # ---------- Candidate restriction (optional) ----------
    def _build_candidate_idx(
        self,
        brand: Optional[str],
        color: Optional[str],
        obj: Optional[str],
        category_name_1: Optional[str] = None,
        category_name_2: Optional[str] = None,
        category_name_3: Optional[str] = None,
        category_any: Optional[str] = None,
    ) -> np.ndarray:
        N = len(self.df)
        mask = np.ones(N, dtype=bool)

        if brand and "brand" in self.df.columns:
            bmask = _norm_eq(self.df["brand"], brand)
            if bmask.any():
                mask &= bmask

        if "category_name_1" in self.df.columns and category_name_1:
            mask &= _norm_eq(self.df["category_name_1"], category_name_1)
        if "category_name_2" in self.df.columns and category_name_2:
            mask &= _norm_eq(self.df["category_name_2"], category_name_2)
        if "category_name_3" in self.df.columns and category_name_3:
            mask &= _norm_eq(self.df["category_name_3"], category_name_3)

        if category_any:
            parts = []
            for col in ["category_name_1", "category_name_2", "category_name_3", "category_name_4"]:
                if col in self.df.columns:
                    parts.append(_norm_eq(self.df[col], category_any))
            if parts:
                anymask = parts[0]
                for p in parts[1:]:
                    anymask = np.logical_or(anymask, p)
                if anymask.any():
                    mask &= anymask

        if obj:
            obj_l = str(obj).strip().lower()
            if obj_l.endswith("es") and not obj_l.endswith("ses"):
                obj_l = obj_l[:-2]
            elif obj_l.endswith("s") and not obj_l.endswith("ss"):
                obj_l = obj_l[:-1]
            toks = {t for t in re.findall(r"[a-z0-9]+", obj_l) if t}
            if toks:
                text = (
                    self.df[self.name_col].fillna("").astype(str) + " " +
                    self.df.get("category_name_1", "").fillna("").astype(str) + " " +
                    self.df.get("category_name_2", "").fillna("").astype(str) + " " +
                    self.df.get("category_name_3", "").fillna("").astype(str) + " " +
                    self.df.get("category_name_4", "").fillna("").astype(str) + " " +
                    self.df.get("request_path_1", "").fillna("").astype(str) + " " +
                    self.df.get("request_path_2", "").fillna("").astype(str) + " " +
                    self.df.get("request_path_3", "").fillna("").astype(str) + " " +
                    self.df.get("request_path_4", "").fillna("").astype(str)
                ).str.lower()
                mask &= text.apply(lambda t: any(tok in t for tok in toks)).to_numpy()

        if color:
            aliases = COLOR_ALIASES.get(color.lower(), {color.lower()})
            text = (
                self.df[self.name_col].fillna("").astype(str) + " " +
                self.df.get("category_name_1", "").fillna("").astype(str) + " " +
                self.df.get("category_name_2", "").fillna("").astype(str) + " " +
                self.df.get("category_name_3", "").fillna("").astype(str) + " " +
                self.df.get("category_name_4", "").fillna("").astype(str)
            ).str.lower()
            mask &= text.apply(lambda t: _contains_any(t, aliases)).to_numpy()

        idx = np.where(mask)[0]
        return idx if idx.size else np.arange(N)

    # ---------- Main search ----------
    def search(
        self,
        query: str,
        *,
        pos_terms: Optional[List[str]] = None,   # three descriptive terms
        top_k: int = 5,
        include_cols: Optional[List[str]] = None,
        candidates_idx: Optional[np.ndarray] = None,
        alpha: float = 0.7,
        biz_weights: Optional[Dict[str, float]] = None,
        brand: Optional[str] = None,
        color: Optional[str] = None,
        object: Optional[str] = None,
        category_name_1: Optional[str] = None,
        category_name_2: Optional[str] = None,
        category_name_3: Optional[str] = None,
        category_any: Optional[str] = None,
    ) -> List[dict]:
        if not query or not query.strip():
            return []

        # Restrict candidate pool if any filters present
        if candidates_idx is None and (
            brand or color or object or category_name_1 or category_name_2 or category_name_3 or category_any
        ):
            candidates_idx = self._build_candidate_idx(
                brand, color, object, category_name_1, category_name_2, category_name_3, category_any
            )

        # --------- Build a single CENTROID vector from 4 components ----------
        # 1) free-text query, 2) category name, 3) object, 4) joined terms
        components: List[str] = []
        if query and query.strip():
            components.append(query)
        if category_any:
            components.append(str(category_any))
        if object:
            components.append(str(object))
        if pos_terms:
            joined_terms = " ".join([t for t in pos_terms if t])
            if joined_terms.strip():
                components.append(joined_terms)
        if not components:
            components = [query]

        q_vecs = [self._encode_text(t) for t in components]
        centroid = q_vecs[0]
        for qv in q_vecs[1:]:
            centroid = centroid + qv
        centroid = centroid * (1.0 / float(len(q_vecs)))

        # If vector space ended up empty, rank by business score only
        if self.X.shape[1] == 0:
            N = len(self.df)
            idx_all = np.arange(N) if candidates_idx is None else candidates_idx
            biz = self._compute_biz(idx_all, biz_weights)
            order = np.argsort(-biz)[: int(top_k)]
            out = self.df.iloc[idx_all[order]].copy()
            out.insert(0, "similarity", 0.0)
            out.insert(1, "business_score", np.round(biz[order], 4))
            out.insert(2, "score", np.round(biz[order], 4))
            return self._finalize(out, include_cols)

        # Cosine similarity once against the centroid vector
        if candidates_idx is None:
            sims = cosine_similarity(centroid, self.X).ravel()
            idx_all = np.arange(self.X.shape[0])
        else:
            sims = cosine_similarity(centroid, self.X[candidates_idx]).ravel()
            idx_all = candidates_idx

        # Business score (only profitability + return_rate)
        biz = self._compute_biz(idx_all, biz_weights)

        # Blend + rank
        alpha = float(np.clip(alpha, 0.0, 1.0))
        score = alpha * sims + (1.0 - alpha) * biz

        order = np.argsort(-score)[: int(top_k)]
        idx = idx_all[order]

        out = self.df.iloc[idx].copy()
        out.insert(0, "similarity", np.round(sims[order], 4))
        out.insert(1, "business_score", np.round(biz[order], 4))
        out.insert(2, "score", np.round(score[order], 4))
        return self._finalize(out, include_cols)

    # ---------- helpers ----------
    def _compute_biz(self, idx_all: np.ndarray, biz_weights: Optional[Dict[str, float]]) -> np.ndarray:
        if self.biz_matrix.size and (biz_weights is not None) and len(biz_weights) > 0:
            w = np.array([biz_weights.get(f, 0.0) for f in self.biz_feature_names], dtype=float)
            if np.allclose(w.sum(), 0.0):
                return np.zeros(len(idx_all))
            w = w / (abs(w).sum())
            biz_full = self.biz_matrix @ w
            return biz_full[idx_all]
        return np.zeros(len(idx_all))

    def _finalize(self, out: pd.DataFrame, include_cols: Optional[List[str]]) -> List[dict]:
        if include_cols is None:
            include_cols = [
                "score", "similarity", "business_score", "product_id",
                self.name_col, "brand", "current_price", "product_url"
            ]
        include_cols = [c for c in include_cols if c in out.columns]
        result = out[include_cols].copy()

        # to native types for JSON
        def _to_native(v):
            return v.item() if isinstance(v, np.generic) else v

        records = result.where(pd.notna(result), None).to_dict(orient="records")
        for r in records:
            for k, v in list(r.items()):
                r[k] = _to_native(v)
        return records
