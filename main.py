"""
main.py
FastAPI service for the Gift Recommendation System.

Loads the artifact produced by Semple_Modeling_and_Evaluation.ipynb
(gift_recommender.joblib: pipeline + KMeans + item embeddings + catalog)
and exposes endpoints to get recommendations and similar items.

Run with:
    uvicorn main:app --reload
"""
from typing import List, Literal

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT_PATH = "gift_recommender.joblib"

artifact = joblib.load(ARTIFACT_PATH)
pipeline = artifact["pipeline"]
kmeans = artifact["kmeans"]
X = artifact["embeddings"]
df = artifact["catalog"]

INTEREST_FLAGS = [c for c in df.columns if c.startswith("int_")]
OCCASION_FLAGS = [c for c in df.columns if c.startswith("occ_")]
INTERESTS = [c.replace("int_", "") for c in INTEREST_FLAGS]
OCCASIONS = [c.replace("occ_", "") for c in OCCASION_FLAGS]

AGE_BUCKETS = [
    ("age_baby", 0, 3),
    ("age_child", 4, 12),
    ("age_teen", 13, 17),
    ("age_youngadult", 18, 29),
    ("age_adult", 30, 49),
    ("age_senior", 50, 99),
]
AGE_BUCKET_COLS = [b[0] for b in AGE_BUCKETS]

PRICE_EDGES = [100, 300, 700, 1500]
PRICE_BAND_COLS = [f"band_{i}" for i in range(5)]

GENDER_COLS = ["serves_female", "serves_male"]
CATEGORICAL_COLS = ["category", "gift_type"]

ALL_FEATURE_COLS = (INTEREST_FLAGS + OCCASION_FLAGS + AGE_BUCKET_COLS
                     + GENDER_COLS + PRICE_BAND_COLS + CATEGORICAL_COLS)


# ---------------------------------------------------------------------------
# Feature engineering (same logic as the notebook, so a query lands in the
# same vector space as the catalog)
# ---------------------------------------------------------------------------
def price_band_vector(price, neighbour_weight=0.5):
    idx = int(np.searchsorted(PRICE_EDGES, price, side="right"))
    v = np.zeros(5, dtype=np.float32)
    v[idx] = 1.0
    if idx - 1 >= 0:
        v[idx - 1] = neighbour_weight
    if idx + 1 < 5:
        v[idx + 1] = neighbour_weight
    return v


def build_query_profile(age, gender, occasion, budget, interests):
    row = {c: 0.0 for c in ALL_FEATURE_COLS if c not in CATEGORICAL_COLS}

    for it in interests:
        key = f"int_{it}"
        if key in row:
            row[key] = 1.0

    okey = f"occ_{occasion}"
    if okey in row:
        row[okey] = 1.0

    for name, b_lo, b_hi in AGE_BUCKETS:
        if b_lo <= age <= b_hi:
            row[name] = 1.0
            break

    if gender == "Female":
        row["serves_female"] = 1.0
    elif gender == "Male":
        row["serves_male"] = 1.0
    else:
        row["serves_female"] = row["serves_male"] = 1.0

    row.update(dict(zip(PRICE_BAND_COLS, price_band_vector(budget))))

    q = pd.DataFrame([row])
    q["category"], q["gift_type"] = "__UNKNOWN__", "__UNKNOWN__"
    return q[ALL_FEATURE_COLS]


def apply_hard_filters(data, age, gender, occasion, budget, tolerance=0.10):
    ceiling = budget * (1 + tolerance)
    mask = (data.min_age <= age) & (data.max_age >= age) & (data.price_min <= ceiling)
    if gender in ("Female", "Male"):
        mask &= data.gender_target.isin([gender, "Unisex"])
    okey = f"occ_{occasion}"
    if okey in data.columns:
        mask &= data[okey].astype(bool)
    return mask.to_numpy()


def recommend(age, gender, occasion, budget, interests, top_k=5):
    q_vec = pipeline.transform(build_query_profile(age, gender, occasion, budget, interests))
    mask = apply_hard_filters(df, age, gender, occasion, budget)
    candidates = np.flatnonzero(mask)

    if len(candidates) == 0:
        return pd.DataFrame(columns=["parent_id", "product_name", "brand",
                                      "category", "price_median", "similarity"])

    sims = X[candidates] @ q_vec.ravel()
    order = np.argsort(-sims)[:top_k]
    top = candidates[order]

    result = df.loc[top, ["parent_id", "product_name", "brand",
                          "category", "price_median"]].copy()
    result["similarity"] = np.round(sims[order], 3)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Gift Recommender", version="1.0.0")


class GiftRequest(BaseModel):
    age: int = Field(..., ge=0, le=99)
    gender: Literal["Female", "Male", "Any"]
    occasion: str
    budget: float = Field(..., gt=0)
    interests: List[str] = []
    top_k: int = Field(5, ge=1, le=50)


@app.get("/options")
def options():
    """Powers UI dropdowns: valid interests / occasions / genders."""
    return {"interests": INTERESTS, "occasions": OCCASIONS,
            "genders": ["Female", "Male", "Any"]}


@app.post("/recommend")
def get_recommendations(req: GiftRequest):
    if req.occasion not in OCCASIONS:
        raise HTTPException(422, f"occasion must be one of {OCCASIONS}")

    recs = recommend(
        age=req.age, gender=req.gender, occasion=req.occasion,
        budget=req.budget, interests=req.interests, top_k=req.top_k,
    )
    return {"recommendations": recs.to_dict(orient="records")}


@app.get("/similar/{parent_id}")
def similar_items(parent_id: str, top_k: int = 5):
    """Item-to-item 'more like this' via cosine similarity on the embeddings."""
    pos = np.flatnonzero(df.parent_id.to_numpy() == parent_id)
    if len(pos) == 0:
        raise HTTPException(404, "parent_id not found")
    pos = int(pos[0])

    sims = X @ X[pos]
    sims[pos] = -1.0
    order = np.argsort(-sims)[:top_k]

    result = df.iloc[order][["parent_id", "product_name", "category", "price_median"]].copy()
    result["similarity"] = np.round(sims[order], 3)
    return result.to_dict(orient="records")
