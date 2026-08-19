# MediTwin AI

**Full title**: MediTwin AI: A Multimodal Multi-Agent Clinical Decision Support System with Explainable AI, Patient Outcome Simulation, and Personalized Treatment Intelligence

**B.Tech AI & Data Science — Review 3 Capstone Submission**

---

## Project Structure

```
MEDI_TWIN_AI/
├── agents/
│   ├── pipeline.py          # LangGraph multi-agent pipeline (5 agents)
│   ├── rag_engine.py        # ChromaDB RAG with 12 medical guideline docs
│   └── fhir_builder.py      # FHIR R4 DiagnosticReport builder
├── backend/
│   ├── main.py              # FastAPI REST API
│   └── config.py            # Centralised settings (LLM provider, paths)
├── data/
│   ├── chest_xray/          # Kaggle Chest X-Ray dataset (train/val/test)
│   └── tabular/             # UCI Heart Disease CSV (auto-downloaded)
├── frontend/
│   └── app.py               # Streamlit 7-tab demo UI
├── models/
│   ├── imaging_model.py     # ResNet-18 + Grad-CAM
│   └── tabular_model.py     # RF+GB ensemble + SHAP
├── scripts/
│   ├── train_models.py      # Training entry point
│   └── generate_report.py   # PDF report generator
├── tests/
│   └── test_meditwin.py     # pytest test suite
├── outputs/                 # Reports, FHIR JSONs, XAI plots
├── .env.example             # Environment config template
├── requirements.txt
└── RESULTS.md               # Generated after training
```

---

## Quick Start

### 1. Create Virtual Environment
```bash
cd MEDI_TWIN_AI
# Use Python 3.12.x for this project on Windows.
# Python 3.13 is not compatible with the pinned NumPy/LangChain stack here.
python3.12 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
# PyTorch CPU (laptop-safe)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# All other dependencies
pip install -r requirements.txt
```

> On Windows, install the official Python 3.12 release first if you do not already have it. The current project pins `numpy==1.26.4`, which is compatible with Python 3.12 but fails on Python 3.13 in this environment.

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env and set your LLM_PROVIDER + API key (groq/openai/anthropic/ollama)
# For Groq (free): get key at https://console.groq.com
```

### 4. Train the Tabular Model (auto-downloads UCI dataset)
```bash
python scripts/train_models.py
```

### 5. Train the Imaging Model (requires Kaggle dataset)
```bash
# Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Extract to: data/chest_xray/  (must have train/val/test folders with NORMAL/PNEUMONIA subfolders)
python scripts/train_models.py --imaging --xray-data data/chest_xray
```

### 6. Run Streamlit Demo
```bash
cd MEDI_TWIN_AI
streamlit run frontend/app.py
```

### 7. Run FastAPI Backend (optional — Streamlit calls pipeline directly)
```bash
uvicorn backend.main:app --reload --port 8000
```

### 8. Run Tests
```bash
pytest tests/ -v
```

---

## Agents

| Agent | Role | XAI |
|---|---|---|
| **Imaging Agent** | ResNet-18 chest X-ray classification | Grad-CAM heatmap |
| **Lab/Risk Agent** | RF+GB cardiac risk prediction | SHAP waterfall + summary |
| **NLP Agent** | Symptom/medication/vitals extraction from notes | Entity highlighting |
| **RAG Agent** | Evidence retrieval from ChromaDB (12 clinical guidelines) | Source citations |
| **Supervisor Agent** | Merges outputs → differential diagnosis + recommendations | Confidence scores |

---

## LLM Configuration

Set `LLM_PROVIDER` in `.env` to one of:

| Provider | Env Var | Free Tier |
|---|---|---|
| `groq` | `GROQ_API_KEY` | ✅ Yes (fast Llama 3.1) |
| `openai` | `OPENAI_API_KEY` | ❌ Paid |
| `anthropic` | `ANTHROPIC_API_KEY` | ❌ Paid |
| `ollama` | *(local)* | ✅ Yes (offline) |

The system works **without any LLM** — it falls back to deterministic rule-based clinical reasoning.

---

## Datasets Used

| Model | Dataset | Source | Size |
|---|---|---|---|
| Imaging | Chest X-Ray Pneumonia | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | 5,863 images |
| Tabular | UCI Heart Disease (Cleveland) | [UCI ML Repo](https://archive.ics.uci.edu/ml/datasets/heart+disease) | 303 samples |
| RAG | Medical guidelines | WHO, AHA/ACC, IDSA, ESC, BTS, RSNA | 12 documents |

---

## Hardware Notes

- **No GPU required** — all models trained on CPU (16 GB RAM sufficient)
- Tabular model training: ~30 seconds
- Imaging model training: ~15-30 min on CPU (10 epochs, batch=16)
- GPU would reduce imaging training to ~3-5 min

---

## FHIR R4 Output

Each analysis generates a valid FHIR R4 `Bundle` containing:
- `DiagnosticReport` — overall clinical findings
- `Observation` (imaging) — CXR classification result
- `Observation` (lab) — cardiac risk prediction  
- `ClinicalImpression` — differential diagnosis + recommendations

Exported as JSON to `outputs/fhir/<run_id>.json` and downloadable from the UI.

---

## Disclaimer

MediTwin AI is an academic research prototype developed for B.Tech capstone evaluation. It is **not a certified medical device** and must not be used for actual clinical diagnosis without validation by licensed healthcare professionals.
