# # backend/app.py
# import os
# from typing import Any, Dict, List, Optional

# import pandas as pd
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field

# from search import CosineSearch

# CSV_PATH = os.getenv("CSV_PATH", "/data/product_catalog.csv")
# NAME_COL = os.getenv("NAME_COL", "name")

# app = FastAPI(title="Cosine Similarity Backend")

# # CORS so Streamlit (different origin) can call the API
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# engine = None
# startup_error = ""
# try:
#     engine = CosineSearch(CSV_PATH, name_col=NAME_COL)
# except Exception as e:
#     startup_error = str(e)


# class SearchRequest(BaseModel):
#     query: str
#     pos_terms: Optional[List[str]] = None     # NEW: three positional terms
#     top_k: int = 5
#     alpha: float = Field(0.7, ge=0.0, le=1.0)
#     biz_weights: Optional[Dict[str, float]] = None
#     brand: Optional[str] = None
#     color: Optional[str] = None
#     object: Optional[str] = None
#     category_name_1: Optional[str] = None
#     category_name_2: Optional[str] = None
#     category_name_3: Optional[str] = None
#     category_any: Optional[str] = None


# class SearchResponse(BaseModel):
#     items: List[Dict[str, Any]]


# # @app.get("/healthz")
# # def healthz():
# #     if engine is None:
# #         return {"status": "degraded", "error": startup_error}
# #     return {
# #         "status": "ok",
# #         "rows": len(engine.df),
# #         "name_col": engine.name_col,
# #         "biz_features": engine.biz_feature_names,
# #         "alpha_default": 0.7,
# #     }

# @app.get("/healthz")
# def healthz():
#     if engine is None:
#         return {"status": "degraded", "error": startup_error}
#     return {
#         "status": "ok",
#         "rows": len(engine.df),
#         "name_col": engine.name_col,
#         "biz_features": engine.biz_feature_names,   # shows ['profitability','return_rate', ...] if found
#         "alpha_default": 0.7,
#     }

# @app.get("/taxonomy")
# def taxonomy():
#     if engine is None:
#         raise HTTPException(status_code=500, detail=f"Engine not ready: {startup_error}")

#     def uniq(col):
#         if col not in engine.df.columns:
#             return []
#         return sorted(
#             [x for x in engine.df[col].dropna().astype(str).str.strip().unique() if x]
#         )

#     return {
#         "category_name_1": uniq("category_name_1"),
#         "category_name_2": uniq("category_name_2"),
#         "category_name_3": uniq("category_name_3"),
#         "brand": uniq("brand")[:2000],
#     }


# @app.post("/search", response_model=SearchResponse)
# def search(req: SearchRequest):
#     if engine is None:
#         raise HTTPException(status_code=500, detail=f"Engine not ready: {startup_error}")
#     items = engine.search(
#         req.query,
#         pos_terms=req.pos_terms,            # <-- pass the three terms
#         top_k=req.top_k,
#         alpha=req.alpha,
#         biz_weights=req.biz_weights,
#         brand=req.brand,
#         color=req.color,
#         object=req.object,
#         category_name_1=req.category_name_1,
#         category_name_2=req.category_name_2,
#         category_name_3=req.category_name_3,
#         category_any=req.category_any,
#     )
#     return {"items": items}






##################### NEW



import os
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from search import CosineSearch

CSV_PATH = os.getenv("CSV_PATH", "/data/product_catalog.csv")
NAME_COL = os.getenv("NAME_COL", "name")

app = FastAPI(title="Cosine Similarity Backend")

# CORS so Streamlit (different origin) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None
startup_error = ""
try:
    engine = CosineSearch(CSV_PATH, name_col=NAME_COL)
except Exception as e:
    startup_error = str(e)


class SearchRequest(BaseModel):
    query: str
    pos_terms: Optional[List[str]] = None
    top_k: int = 5
    alpha: float = Field(0.7, ge=0.0, le=1.0)
    biz_weights: Optional[Dict[str, float]] = None
    brand: Optional[str] = None
    color: Optional[str] = None
    object: Optional[str] = None
    category_name_1: Optional[str] = None
    category_name_2: Optional[str] = None
    category_name_3: Optional[str] = None
    category_any: Optional[str] = None


class SearchResponse(BaseModel):
    items: List[Dict[str, Any]]


@app.get("/healthz")
def healthz():
    if engine is None:
        return {"status": "degraded", "error": startup_error}
    return {
        "status": "ok",
        "rows": len(engine.df),
        "name_col": engine.name_col,
        "biz_features": engine.biz_feature_names,   # e.g. ['profitability','return_rate']
        "alpha_default": 0.7,
    }


@app.get("/taxonomy")
def taxonomy():
    if engine is None:
        raise HTTPException(status_code=500, detail=f"Engine not ready: {startup_error}")

    def uniq(col):
        if col not in engine.df.columns:
            return []
        return sorted(
            [x for x in engine.df[col].dropna().astype(str).str.strip().unique() if x]
        )

    return {
        "category_name_1": uniq("category_name_1"),
        "category_name_2": uniq("category_name_2"),
        "category_name_3": uniq("category_name_3"),
        "brand": uniq("brand")[:2000],
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail=f"Engine not ready: {startup_error}")
    items = engine.search(
        req.query,
        pos_terms=req.pos_terms,
        top_k=req.top_k,
        alpha=req.alpha,
        biz_weights=req.biz_weights,
        brand=req.brand,
        color=req.color,
        object=req.object,
        category_name_1=req.category_name_1,
        category_name_2=req.category_name_2,
        category_name_3=req.category_name_3,
        category_any=req.category_any,
    )
    return {"items": items}
