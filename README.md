# ⚛ Quantum Policy Search

A web-based **Quantum-Inspired Policy Search** platform that allows users to explore and analyze **education policies** using quantum-style embeddings and interactive visual charts.

---

## 🚀 What This App Does
- Lets users search education policies using natural language.
- Uses **quantum-inspired semantic embeddings** (tanh + sine hybrid vectors).
- Displays **interactive insights** using Chart.js:
  - 📈 Relevance Score (Line Chart)
  - 🌍 Region-wise Distribution (Pie Chart)
  - 📅 Year-wise Policy Count (Bar Chart)

---

## 🧠 Tech Stack
- **Backend:** FastAPI  
- **Frontend:** Jinja2 + Tailwind CSS + Chart.js  
- **ML:** TF‑IDF, PCA, Scaling (scikit‑learn)

---

## 📁 Folder Structure

```
QuantumPolicySearch/
│
├── app.py
├── train_quantum.py
├── quantum_model.pkl
├── education_policies.csv
│
├── templates/
│   └── quantum.html
│
└── static/
```

---

## ⚙️ Installation Guide

### 1️⃣ Create Virtual Environment
```
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Train Model
```
python train_quantum.py
```
This generates **quantum_model.pkl**.

### 4️⃣ Run App
```
uvicorn app:app --reload
```
Visit: **http://127.0.0.1:8000**

---

## 📊 Dataset Requirements
Your CSV must contain at least:

- `title`
- `region`
- `year`
- `summary` or `description` or `full_text`

---

## 🧾 Troubleshooting

| Issue | Fix |
|------|-----|
| NumPy installation error | Use **Python 3.11** |
| Charts not showing | Inspect browser Console (F12 → Console) |
| Template error | Ensure `quantum.html` is inside `templates/` |
| No results | Try a broader query |

---

## 👨‍💻 Author
Developed by **Yateesh V Mattur**

Powered by **Quantum Intelligence ⚛**

---

## 🏁 License
Open-source — for education & research.
