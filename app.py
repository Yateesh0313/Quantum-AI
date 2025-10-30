# ---------- app.py ----------
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Load Quantum Model
# -----------------------------
model_data = joblib.load("quantum_model.pkl")
vectorizer = model_data["vectorizer"]
scaler = model_data["scaler"]
pca = model_data["pca"]
embeddings = model_data["embeddings"]
df = model_data["df"]

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("quantum.html", {"request": request})


@app.post("/quantum_search", response_class=HTMLResponse)
async def quantum_search(request: Request, query: str = Form(...)):
    query_vec = vectorizer.transform([query]).toarray()
    query_scaled = scaler.transform(query_vec)
    query_pca = pca.transform(query_scaled)
    query_encoded = np.tanh(query_pca * np.pi)
    query_quantum = np.concatenate([query_encoded, np.sin(query_encoded)], axis=1)

    similarities = cosine_similarity(query_quantum, embeddings)[0]
    df["score"] = similarities
    results = df.sort_values(by="score", ascending=False).head(6).fillna("N/A")

    region_data = (
        results["region"].value_counts().to_dict()
        if "region" in results.columns else {"Unknown": len(results)}
    )
    status_data = (
        results["status"].value_counts().to_dict()
        if "status" in results.columns else {"Unspecified": len(results)}
    )
    scores_data = dict(zip(results["title"], results["score"]))

    return templates.TemplateResponse("quantum.html", {
        "request": request,
        "query": query,
        "results": results.to_dict(orient="records"),
        "region_data": region_data,
        "status_data": status_data,
        "scores_data": scores_data
    })
