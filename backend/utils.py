# backend/utils.py
from __future__ import annotations
from typing import Iterable, Mapping, Optional, Union, Dict
import json, numpy as np, pandas as pd

def topn_df_to_json_map(
    df: pd.DataFrame,
    n: int = 10,
    *,
    key_col: str = "product_id",
    include: Iterable[str] = ("name", "current_price"),
    rename: Optional[Mapping[str, str]] = {"current_price": "price"},
    as_string: bool = False,
    write_path: Optional[str] = None,
) -> Union[Dict, str]:
    cols = [key_col] + [c for c in include if c in df.columns]
    if key_col not in df.columns:
        raise ValueError(f"Missing key_col '{key_col}' in DataFrame.")
    if len(cols) == 1:
        raise ValueError("No value columns found from `include` in DataFrame.")
    sub = df[cols].head(n).copy()
    if rename:
        sub = sub.rename(columns=rename)
    sub = sub.where(pd.notna(sub), None)
    def _to_native(v):
        return v.item() if isinstance(v, np.generic) else v
    sub = sub.applymap(_to_native)
    out = {str(row[key_col]): {k: row[k] for k in sub.columns if k != key_col} for _, row in sub.iterrows()}
    if write_path:
        with open(write_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return json.dumps(out, ensure_ascii=False, indent=2) if as_string else out
