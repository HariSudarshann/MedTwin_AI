"""
MediTwin AI — Automated Test Suite
Run: pytest tests/ -v
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np


# ── Test: Tabular Model ───────────────────────────────────────────────────────
class TestTabularModel:
    def test_download_dataset(self):
        from models.tabular_model import download_heart_disease_dataset
        df = download_heart_disease_dataset(ROOT / "data")
        assert len(df) > 0, "Dataset should have rows"
        assert "target" in df.columns
        assert len(df.columns) == 14

    def test_preprocess(self):
        from models.tabular_model import download_heart_disease_dataset, preprocess_heart_data
        df = download_heart_disease_dataset(ROOT / "data")
        X, y = preprocess_heart_data(df)
        assert X.shape[1] == 13
        assert set(y.unique()).issubset({0, 1})

    def test_training(self, tmp_path):
        from models.tabular_model import train_tabular_model
        save_path = tmp_path / "test_model.joblib"
        metrics = train_tabular_model(ROOT / "data", save_path)
        assert save_path.exists()
        assert metrics["accuracy"] > 0.5
        assert metrics["roc_auc"] > 0.5
        assert 0 <= metrics["f1"] <= 1

    def test_inference(self, tmp_path):
        from models.tabular_model import train_tabular_model, LabRiskClassifier
        save_path = tmp_path / "test_model.joblib"
        train_tabular_model(ROOT / "data", save_path)
        clf = LabRiskClassifier(save_path)
        lab_values = {
            "age": 55, "sex": 1, "cp": 2, "trestbps": 145, "chol": 240,
            "fbs": 0, "restecg": 1, "thalach": 148, "exang": 0,
            "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2
        }
        result = clf.predict(lab_values)
        assert result["prediction"] in ("LOW RISK", "HIGH RISK")
        assert 0 <= result["confidence"] <= 1
        assert "shap_waterfall_b64" in result
        assert len(result["top_risk_factors"]) > 0


# ── Test: RAG Engine ──────────────────────────────────────────────────────────
class TestRAGEngine:
    def test_keyword_fallback(self):
        from agents.rag_engine import _keyword_fallback
        results = _keyword_fallback("pneumonia chest xray", top_k=3)
        assert len(results) > 0
        assert "title" in results[0]
        assert "source" in results[0]
        assert "snippet" in results[0]

    def test_corpus_content(self):
        from agents.rag_engine import MEDICAL_GUIDELINES_CORPUS
        assert len(MEDICAL_GUIDELINES_CORPUS) >= 10
        for doc in MEDICAL_GUIDELINES_CORPUS:
            assert "id" in doc
            assert "title" in doc
            assert "content" in doc
            assert len(doc["content"]) > 50


# ── Test: FHIR Builder ────────────────────────────────────────────────────────
class TestFHIRBuilder:
    def get_mock_state(self):
        return {
            "patient_id": "TEST-001",
            "imaging_result": {"prediction": "PNEUMONIA", "confidence": 0.87},
            "lab_result": {"prediction": "HIGH RISK", "confidence": 0.73, "top_risk_factors": [("age", 0.15)]},
            "nlp_result": {"symptoms": ["cough", "fever"], "conditions": ["hypertension"]},
            "rag_evidence": [{"rank": 1, "title": "WHO Pneumonia Guidelines",
                              "source": "WHO 2023", "snippet": "..."}],
            "final_report": {
                "summary": "Test summary",
                "overall_confidence": 0.80,
                "differential_diagnosis": [{"diagnosis": "CAP", "likelihood": "HIGH", "basis": "CXR"}],
                "recommendations": ["Start antibiotics"],
            }
        }

    def test_fhir_bundle_structure(self):
        from agents.fhir_builder import build_fhir_report
        state = self.get_mock_state()
        bundle = build_fhir_report(state)
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert len(bundle["entry"]) == 4
        types = [e["resource"]["resourceType"] for e in bundle["entry"]]
        assert "DiagnosticReport" in types
        assert "Observation" in types
        assert "ClinicalImpression" in types

    def test_fhir_json_serializable(self):
        from agents.fhir_builder import build_fhir_report
        bundle = build_fhir_report(self.get_mock_state())
        json_str = json.dumps(bundle, default=str)
        assert len(json_str) > 100


# ── Test: NLP Agent ───────────────────────────────────────────────────────────
class TestNLPAgent:
    def test_symptom_extraction(self):
        from agents.pipeline import nlp_agent
        state = {
            "patient_id": "T1",
            "clinical_notes": "Patient has fever, cough, and shortness of breath. BP: 145/90.",
            "errors": []
        }
        result = nlp_agent(state)
        nlp = result["nlp_result"]
        assert "fever" in nlp["symptoms"]
        assert "cough" in nlp["symptoms"]
        assert "shortness of breath" in nlp["symptoms"]
        assert nlp["vitals"].get("bp_systolic") == 145

    def test_medication_extraction(self):
        from agents.pipeline import nlp_agent
        state = {
            "patient_id": "T2",
            "clinical_notes": "On metformin 500mg and aspirin 75mg. History of hypertension.",
            "errors": []
        }
        result = nlp_agent(state)
        nlp = result["nlp_result"]
        meds = [m.lower() for m in nlp["medications"]]
        assert "metformin" in meds
        assert "aspirin" in meds
        assert "hypertension" in nlp["conditions"]


# ── Test: Full Pipeline (no models) ──────────────────────────────────────────
class TestPipeline:
    def test_pipeline_runs_without_models(self):
        from agents.pipeline import run_pipeline
        report = run_pipeline(
            patient_id="TEST-NOOP",
            xray_bytes=None,
            lab_values={"age": 55, "sex": 1, "cp": 2, "trestbps": 140,
                        "chol": 220, "fbs": 0, "restecg": 1, "thalach": 150,
                        "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2},
            clinical_notes="Patient with fever and cough. BP: 130/85.",
        )
        assert isinstance(report, dict)
        assert "summary" in report or "differential_diagnosis" in report

    def test_pipeline_nlp_always_runs(self):
        from agents.pipeline import run_pipeline
        report = run_pipeline(
            patient_id="TEST-NLP",
            clinical_notes="Fever, cough, chest pain. HR: 102 bpm.",
            lab_values={},
        )
        nlp = report.get("nlp_result", {})
        assert "fever" in nlp.get("symptoms", [])


# ── Test: API Endpoints ───────────────────────────────────────────────────────
class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "MediTwin" in r.json()["name"]

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "imaging_model" in data
        assert "tabular_model" in data

    def test_analyze_no_image(self, client):
        r = client.post("/api/analyze", data={
            "patient_id": "API-TEST",
            "clinical_notes": "Fever and cough for 3 days.",
            "lab_values_json": json.dumps({
                "age": 50, "sex": 1, "cp": 1, "trestbps": 130,
                "chol": 210, "fbs": 0, "restecg": 0, "thalach": 155,
                "exang": 0, "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 2
            })
        })
        assert r.status_code == 200
        data = r.json()
        assert "run_id" in data
        assert "report" in data
