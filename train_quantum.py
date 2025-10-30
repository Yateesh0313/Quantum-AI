# ---------- train_quantum.py ----------
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------
# Step 1: Load Dataset (.csv)
# -----------------------------
print("📂 Loading dataset...")
df = pd.read_csv("education_policies.csv")

# Basic cleaning
text_columns = [c for c in df.columns if df[c].dtype == 'object']
for col in text_columns:
    df[col] = df[col].fillna('').astype(str)

# Prefer a main text column
if 'summary' in df.columns:
    df["text"] = df["summary"]
elif 'description' in df.columns:
    df["text"] = df["description"]
elif 'full_text' in df.columns:
    df["text"] = df["full_text"]
else:
    df["text"] = df.apply(lambda x: " ".join(x.astype(str)), axis=1)

print(f"✅ Dataset loaded with {len(df)} rows")

# -----------------------------
# Step 2: Vectorization (TF-IDF)
# -----------------------------
print("⚙️ Generating TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer.fit_transform(df["text"]).toarray()

# -----------------------------
# Step 3: Scaling and PCA
# -----------------------------
print("📊 Scaling and reducing dimensions...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_tfidf)

pca = PCA(n_components=8, random_state=42)
X_reduced = pca.fit_transform(X_scaled)

# -----------------------------
# Step 4: Quantum-inspired Embeddings
# -----------------------------
print("🧠 Creating quantum-inspired embeddings...")
quantum_embeddings = []
for vec in X_reduced:
    encoded = np.tanh(vec * np.pi)
    quantum_vec = np.concatenate([encoded, np.sin(encoded)])  # → 16D vector
    quantum_embeddings.append(quantum_vec)

quantum_embeddings = np.array(quantum_embeddings)
print(f"✅ Generated quantum embeddings with shape {quantum_embeddings.shape}")

# -----------------------------
# Step 5: Save Model Data
# -----------------------------
print("💾 Saving model...")
model_data = {
    "vectorizer": vectorizer,
    "scaler": scaler,
    "pca": pca,
    "embeddings": quantum_embeddings,
    "df": df
}
joblib.dump(model_data, "quantum_model.pkl")

print("🎉 Training complete! Model saved as quantum_model.pkl")
