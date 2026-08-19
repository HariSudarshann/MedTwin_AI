"""
MediTwin AI — FastAPI Backend
"""
import json
import io
import uuid
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MediTwin AI API",
    description="Multimodal Multi-Agent Clinical Decision Support System",
    version="3.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── SQLite logging ────────────────────────────────────────────────────────────
DB_PATH = Path("outputs/meditwin_runs.db")

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            patient_id TEXT,
            timestamp TEXT,
            xray_prediction TEXT,
            xray_confidence REAL,
            lab_prediction TEXT,
            lab_confidence REAL,
            overall_confidence REAL,
            report_json TEXT
        )
    """)
    conn.commit(); conn.close()

def log_run(run_id: str, patient_id: str, report: Dict):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("""
            INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            run_id, patient_id, datetime.utcnow().isoformat(),
            report.get("imaging_result", {}).get("prediction", "N/A"),
            report.get("imaging_result", {}).get("confidence", 0),
            report.get("lab_result", {}).get("prediction", "N/A"),
            report.get("lab_result", {}).get("confidence", 0),
            report.get("overall_confidence", 0),
            json.dumps(report, default=str)
        ))
        conn.commit(); conn.close()
    except Exception as e:
        logger.warning(f"DB log failed: {e}")

init_db()


# ── Request Models ────────────────────────────────────────────────────────────
class LabRequest(BaseModel):
    patient_id: str = "DEMO-001"
    lab_values: Dict[str, float]
    clinical_notes: str = ""


class AnalysisRequest(BaseModel):
    patient_id: str = "DEMO-001"
    lab_values: Optional[Dict[str, float]] = None
    clinical_notes: str = ""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"name": "MediTwin AI", "version": "3.0.0", "status": "running"}

@app.get("/health")
def health():
    xray_model_ready = Path("models/resnet18_xray.pt").exists()
    tab_model_ready  = Path("models/heart_risk_rf.joblib").exists()
    return {
        "status": "ok",
        "imaging_model": "ready" if xray_model_ready else "not_trained",
        "tabular_model": "ready" if tab_model_ready else "not_trained",
    }


@app.post("/api/analyze")
async def analyze(
    patient_id: str = Form("DEMO-001"),
    clinical_notes: str = Form(""),
    lab_values_json: str = Form("{}"),
    xray_file: Optional[UploadFile] = File(None),
):
    """Full multimodal analysis — all agents."""
    run_id = str(uuid.uuid4())
    try:
        lab_values = json.loads(lab_values_json) if lab_values_json else {}
    except Exception:
        lab_values = {}

    xray_bytes = None
    if xray_file and xray_file.filename:
        xray_bytes = await xray_file.read()

    report = run_pipeline(
        patient_id=patient_id,
        xray_bytes=xray_bytes,
        lab_values=lab_values,
        clinical_notes=clinical_notes,
    )

    log_run(run_id, patient_id, report)

    # Save FHIR JSON
    fhir_path = Path(f"outputs/fhir/{run_id}.json")
    fhir_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fhir_path, "w") as f:
        json.dump(report.get("fhir_report", {}), f, indent=2, default=str)

    return JSONResponse(content={"run_id": run_id, "report": report}, status_code=200)


@app.post("/api/predict/xray")
async def predict_xray(xray_file: UploadFile = File(...)):
    """Imaging-only endpoint."""
    model_path = Path("models/resnet18_xray.pt")
    if not model_path.exists():
        raise HTTPException(503, "Imaging model not trained. Run train_models.py first.")
    try:
        from models.imaging_model import XRayClassifier
        from PIL import Image
        clf = XRayClassifier(model_path)
        img_bytes = await xray_file.read()
        image = Image.open(io.BytesIO(img_bytes))
        result = clf.predict(image)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/predict/lab")
def predict_lab(req: LabRequest):
    """Lab risk only endpoint."""
    model_path = Path("models/heart_risk_rf.joblib")
    if not model_path.exists():
        raise HTTPException(503, "Tabular model not trained. Run train_models.py first.")
    try:
        from models.tabular_model import LabRiskClassifier
        clf = LabRiskClassifier(model_path)
        result = clf.predict(req.lab_values)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/fhir/{run_id}")
def get_fhir(run_id: str):
    path = Path(f"outputs/fhir/{run_id}.json")
    if not path.exists():
        raise HTTPException(404, "FHIR report not found")
    return FileResponse(path, media_type="application/fhir+json")


@app.get("/api/runs")
def list_runs(limit: int = 20):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            "SELECT run_id, patient_id, timestamp, xray_prediction, lab_prediction, overall_confidence FROM runs ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [{"run_id": r[0], "patient_id": r[1], "timestamp": r[2],
                 "xray": r[3], "lab": r[4], "confidence": r[5]} for r in rows]
    except Exception as e:
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
