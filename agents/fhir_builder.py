"""
MediTwin AI — FHIR R4 Report Builder
Generates FHIR R4-compliant DiagnosticReport + Observation JSONs.
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any


def build_fhir_report(state: Dict) -> Dict:
    """
    Build a FHIR R4 Bundle containing:
    - DiagnosticReport (primary)
    - Observation (imaging result)
    - Observation (lab/risk result)
    - ClinicalImpression (differential + recommendations)
    """
    patient_id = state.get("patient_id", "UNKNOWN")
    now_iso = datetime.now(timezone.utc).isoformat()
    report_id = str(uuid.uuid4())
    imaging = state.get("imaging_result", {})
    lab     = state.get("lab_result", {})
    nlp     = state.get("nlp_result", {})
    evidence= state.get("rag_evidence", [])

    # ── Observations ─────────────────────────────────────────────────────────
    imaging_obs = _build_imaging_observation(patient_id, imaging, now_iso)
    lab_obs     = _build_lab_observation(patient_id, lab, now_iso)

    # ── Clinical Impression ───────────────────────────────────────────────────
    final = state.get("final_report", {})
    differentials = final.get("differential_diagnosis", [])
    clinical_impression = {
        "resourceType": "ClinicalImpression",
        "id": str(uuid.uuid4()),
        "status": "completed",
        "subject": {"reference": f"Patient/{patient_id}"},
        "date": now_iso,
        "description": final.get("summary", ""),
        "finding": [
            {
                "itemCodeableConcept": {
                    "text": d["diagnosis"]
                },
                "basis": f"{d.get('likelihood', 'N/A')} — {d.get('basis', '')}"
            }
            for d in differentials
        ],
        "note": [{"text": rec} for rec in final.get("recommendations", [])],
    }

    # ── DiagnosticReport ──────────────────────────────────────────────────────
    diagnostic_report = {
        "resourceType": "DiagnosticReport",
        "id": report_id,
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                                   "code": "RAD", "display": "Radiology"}]}],
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "24627-2",
                        "display": "Chest X-ray 2 views"}],
            "text": "MediTwin AI — Multimodal Clinical Analysis"
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": now_iso,
        "issued": now_iso,
        "result": [
            {"reference": f"Observation/{imaging_obs['id']}"},
            {"reference": f"Observation/{lab_obs['id']}"},
        ],
        "conclusion": final.get("summary", ""),
        "conclusionCode": [
            {"coding": [{"system": "http://snomed.info/sct",
                         "code": "233604007" if imaging.get("prediction") == "PNEUMONIA" else "162668002",
                         "display": imaging.get("prediction", "Not determined")}]}
        ],
        "extension": [
            {
                "url": "https://meditwin.ai/fhir/extension/ai-confidence",
                "valueDecimal": final.get("overall_confidence", 0)
            },
            {
                "url": "https://meditwin.ai/fhir/extension/evidence-citations",
                "valueString": json.dumps([
                    {"title": e.get("title"), "source": e.get("source")}
                    for e in evidence[:3]
                ])
            }
        ]
    }

    # ── Bundle ────────────────────────────────────────────────────────────────
    bundle = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "collection",
        "timestamp": now_iso,
        "meta": {
            "profile": ["https://meditwin.ai/fhir/StructureDefinition/meditwin-report"],
            "tag": [{"system": "https://meditwin.ai/fhir/tags", "code": "ai-generated",
                     "display": "AI-Generated — Requires Clinician Verification"}]
        },
        "entry": [
            {"resource": diagnostic_report},
            {"resource": imaging_obs},
            {"resource": lab_obs},
            {"resource": clinical_impression},
        ]
    }
    return bundle


def _build_imaging_observation(patient_id, imaging, timestamp) -> Dict:
    pred = imaging.get("prediction", "N/A")
    conf = imaging.get("confidence", 0)
    return {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                   "code": "imaging", "display": "Imaging"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": "24627-2",
                              "display": "CXR AI Classification"}]},
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": timestamp,
        "valueCodeableConcept": {"coding": [{"display": pred}], "text": pred},
        "interpretation": [{"coding": [{"display": f"AI Confidence: {conf:.1%}"}]}],
        "component": [
            {"code": {"text": "AI Prediction"}, "valueString": pred},
            {"code": {"text": "Confidence Score"}, "valueDecimal": round(conf, 4)},
        ]
    }


def _build_lab_observation(patient_id, lab, timestamp) -> Dict:
    pred = lab.get("prediction", "N/A")
    conf = lab.get("confidence", 0)
    top_factors = lab.get("top_risk_factors", [])
    return {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                   "code": "laboratory", "display": "Laboratory"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": "72166-2",
                              "display": "Cardiac Risk Assessment"}]},
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": timestamp,
        "valueCodeableConcept": {"coding": [{"display": pred}], "text": pred},
        "interpretation": [{"coding": [{"display": f"AI Confidence: {conf:.1%}"}]}],
        "component": [
            {"code": {"text": "Risk Prediction"}, "valueString": pred},
            {"code": {"text": "Confidence"}, "valueDecimal": round(conf, 4)},
            {"code": {"text": "Top SHAP Factors"},
             "valueString": str(top_factors[:3])},
        ]
    }
