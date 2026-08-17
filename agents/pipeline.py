"""
MediTwin AI — LangGraph Multi-Agent Pipeline
Agents: Imaging, Lab/Risk, NLP, RAG, Supervisor
"""
import json
import logging
import re
from typing import TypedDict, Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# ── State Schema ─────────────────────────────────────────────────────────────
class ClinicalState(TypedDict, total=False):
    patient_id: str
    clinical_notes: str
    lab_values: Dict[str, float]
    xray_image_bytes: Optional[bytes]
    xray_image_path: Optional[str]

    # Agent outputs
    imaging_result: Optional[Dict]
    lab_result: Optional[Dict]
    nlp_result: Optional[Dict]
    rag_evidence: Optional[List[Dict]]

    # Supervisor final output
    final_report: Optional[Dict]
    errors: List[str]


# ── LLM Factory (swappable provider) ─────────────────────────────────────────
def get_llm(provider: str = "groq", **kwargs):
    """Return a LangChain-compatible LLM based on config."""
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            import os
            return ChatOpenAI(model=kwargs.get("model", "gpt-4o-mini"),
                              api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            import os
            return ChatAnthropic(model=kwargs.get("model", "claude-3-haiku-20240307"),
                                 api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0)
        elif provider == "groq":
            from langchain_groq import ChatGroq
            import os
            return ChatGroq(model=kwargs.get("model", "llama-3.1-70b-versatile"),
                            api_key=os.getenv("GROQ_API_KEY"), temperature=0)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            import os
            return ChatOllama(model=kwargs.get("model", "llama3.1:8b"),
                              base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    except Exception as e:
        logger.warning(f"LLM provider '{provider}' failed: {e}. Falling back to rule-based reasoning.")
    return None


# ── Imaging Agent ─────────────────────────────────────────────────────────────
def imaging_agent(state: ClinicalState) -> ClinicalState:
    """Run ResNet-18 Chest X-Ray classifier + Grad-CAM."""
    from pathlib import Path as P
    model_path = P("models/resnet18_xray.pt")
    errors = state.get("errors", [])

    if not model_path.exists():
        logger.warning("Imaging model not found — skipping imaging agent")
        state["imaging_result"] = {"prediction": "N/A", "confidence": 0,
                                    "note": "Model not trained yet. Run train_models.py first."}
        return state

    try:
        from models.imaging_model import XRayClassifier
        from PIL import Image
        import io

        clf = XRayClassifier(model_path)
        img_bytes = state.get("xray_image_bytes")
        img_path  = state.get("xray_image_path")

        if img_bytes:
            image = Image.open(io.BytesIO(img_bytes))
        elif img_path and Path(img_path).exists():
            image = Image.open(img_path)
        else:
            state["imaging_result"] = {"prediction": "N/A", "confidence": 0,
                                        "note": "No X-ray image provided"}
            return state

        result = clf.predict(image)
        state["imaging_result"] = result
        logger.info(f"[ImagingAgent] {result['prediction']} ({result['confidence']:.2%})")
    except Exception as e:
        errors.append(f"ImagingAgent error: {str(e)}")
        state["imaging_result"] = {"prediction": "ERROR", "confidence": 0, "error": str(e)}
        state["errors"] = errors
    return state


# ── Lab/Risk Agent ────────────────────────────────────────────────────────────
def lab_risk_agent(state: ClinicalState) -> ClinicalState:
    """Run RandomForest cardiac risk classifier + SHAP."""
    from pathlib import Path as P
    model_path = P("models/heart_risk_rf.joblib")
    errors = state.get("errors", [])

    lab_values = state.get("lab_values", {})
    if not lab_values:
        state["lab_result"] = {"prediction": "N/A", "note": "No lab values provided"}
        return state

    if not model_path.exists():
        state["lab_result"] = {"prediction": "N/A", "note": "Model not trained yet. Run train_models.py first."}
        return state

    try:
        from models.tabular_model import LabRiskClassifier
        clf = LabRiskClassifier(model_path)
        result = clf.predict(lab_values)
        state["lab_result"] = result
        logger.info(f"[LabAgent] {result['prediction']} ({result['confidence']:.2%})")
    except Exception as e:
        errors.append(f"LabAgent error: {str(e)}")
        state["lab_result"] = {"prediction": "ERROR", "error": str(e)}
        state["errors"] = errors
    return state


# ── NLP / Clinical Notes Agent ────────────────────────────────────────────────
def nlp_agent(state: ClinicalState) -> ClinicalState:
    """Extract clinical entities from notes using regex + optional transformers."""
    notes = state.get("clinical_notes", "")
    if not notes:
        state["nlp_result"] = {"entities": {}, "note": "No clinical notes provided"}
        return state

    notes_lower = notes.lower()
    symptoms_vocab = {
        "chest pain": ["chest pain", "chest tightness", "angina"],
        "shortness of breath": ["dyspnea", "shortness of breath", "breathlessness", "sob"],
        "fever": ["fever", "pyrexia", "febrile", "temperature"],
        "cough": ["cough", "productive cough", "dry cough"],
        "fatigue": ["fatigue", "weakness", "lethargy", "malaise"],
        "palpitations": ["palpitations", "irregular heartbeat"],
        "syncope": ["syncope", "fainting", "loss of consciousness"],
        "edema": ["edema", "swelling", "oedema"],
        "diaphoresis": ["diaphoresis", "sweating", "sweats"],
        "nausea": ["nausea", "vomiting", "nauseous"],
    }
    conditions_vocab = ["diabetes", "hypertension", "copd", "asthma", "heart failure",
                        "coronary artery disease", "atrial fibrillation", "pneumonia",
                        "tuberculosis", "cancer", "ckd", "renal failure"]
    medications_patterns = r'\b(metformin|aspirin|atorvastatin|lisinopril|amlodipine|warfarin|heparin|insulin|furosemide|metoprolol|amoxicillin|azithromycin|ceftriaxone)\b'

    found_symptoms = [sym for sym, kws in symptoms_vocab.items() if any(kw in notes_lower for kw in kws)]
    found_conditions = [c for c in conditions_vocab if c in notes_lower]
    found_meds = re.findall(medications_patterns, notes_lower, re.IGNORECASE)

    # Extract vitals
    vitals = {}
    bp_match = re.search(r'bp[:\s]*(\d{2,3})[/\\](\d{2,3})', notes_lower)
    if bp_match:
        vitals["bp_systolic"] = int(bp_match.group(1))
        vitals["bp_diastolic"] = int(bp_match.group(2))
    hr_match = re.search(r'(?:hr|heart rate|pulse)[:\s]*(\d{2,3})', notes_lower)
    if hr_match:
        vitals["heart_rate"] = int(hr_match.group(1))
    temp_match = re.search(r'(?:temp|temperature)[:\s]*(\d{2,3}\.?\d*)', notes_lower)
    if temp_match:
        vitals["temperature"] = float(temp_match.group(1))
    o2_match = re.search(r'(?:spo2|o2 sat|oxygen)[:\s]*(\d{2,3})', notes_lower)
    if o2_match:
        vitals["spo2"] = int(o2_match.group(1))

    state["nlp_result"] = {
        "symptoms": found_symptoms,
        "conditions": found_conditions,
        "medications": list(set(found_meds)),
        "vitals": vitals,
        "note_length": len(notes),
    }
    logger.info(f"[NLPAgent] Symptoms={found_symptoms}, Conditions={found_conditions}")
    return state


# ── RAG / Evidence Retrieval Agent ───────────────────────────────────────────
def rag_agent(state: ClinicalState) -> ClinicalState:
    """Retrieve evidence from vector DB based on clinical context."""
    from pathlib import Path as P
    errors = state.get("errors", [])

    imaging_result = state.get("imaging_result", {})
    lab_result     = state.get("lab_result", {})
    nlp_result     = state.get("nlp_result", {})

    # Build composite query
    parts = []
    pred = imaging_result.get("prediction", "")
    if pred and pred not in ("N/A", "ERROR"):
        parts.append(f"chest xray {pred.lower()}")
    risk = lab_result.get("prediction", "")
    if risk and risk not in ("N/A", "ERROR"):
        parts.append(f"cardiac risk {risk.lower()}")
    if nlp_result.get("symptoms"):
        parts.append(" ".join(nlp_result["symptoms"][:3]))
    if nlp_result.get("conditions"):
        parts.append(" ".join(nlp_result["conditions"][:2]))

    query = " ".join(parts) if parts else "clinical diagnosis treatment guidelines"

    try:
        from agents.rag_engine import build_rag_corpus, retrieve_evidence
        persist_dir = P("outputs/chromadb")
        collection = build_rag_corpus(persist_dir)
        evidence = retrieve_evidence(query, collection, top_k=3)
        state["rag_evidence"] = evidence
        logger.info(f"[RAGAgent] Retrieved {len(evidence)} evidence docs for query: '{query}'")
    except Exception as e:
        errors.append(f"RAGAgent error: {str(e)}")
        from agents.rag_engine import _keyword_fallback
        state["rag_evidence"] = _keyword_fallback(query, top_k=3)
        state["errors"] = errors
    return state


# ── Supervisor Agent ──────────────────────────────────────────────────────────
def supervisor_agent(state: ClinicalState) -> ClinicalState:
    """Merge all agent outputs → final clinical report + FHIR JSON."""
    imaging = state.get("imaging_result", {})
    lab     = state.get("lab_result", {})
    nlp     = state.get("nlp_result", {})
    evidence= state.get("rag_evidence", [])

    # ── Build clinical summary ────────────────────────────────────────────────
    summary_parts = []

    xray_pred = imaging.get("prediction", "N/A")
    xray_conf = imaging.get("confidence", 0)
    if xray_pred not in ("N/A", "ERROR"):
        summary_parts.append(
            f"Chest X-Ray analysis: **{xray_pred}** (confidence: {xray_conf:.1%}). "
        )

    lab_pred = lab.get("prediction", "N/A")
    lab_conf = lab.get("confidence", 0)
    if lab_pred not in ("N/A", "ERROR"):
        top_factors = lab.get("top_risk_factors", [])
        factor_str = ", ".join([f"{f} (SHAP={v:+.3f})" for f, v in top_factors[:3]])
        summary_parts.append(
            f"Cardiac risk assessment: **{lab_pred}** (confidence: {lab_conf:.1%}). "
            f"Key contributing factors: {factor_str}."
        )

    if nlp.get("symptoms"):
        summary_parts.append(f"Reported symptoms: {', '.join(nlp['symptoms'])}.")
    if nlp.get("conditions"):
        summary_parts.append(f"Relevant history: {', '.join(nlp['conditions'])}.")

    # ── Differential Diagnosis ────────────────────────────────────────────────
    differentials = _generate_differentials(xray_pred, lab_pred, nlp)

    # ── Treatment Recommendations ─────────────────────────────────────────────
    recommendations = _generate_recommendations(xray_pred, lab_pred, nlp)

    # ── Overall Confidence ────────────────────────────────────────────────────
    confs = [c for c in [xray_conf, lab_conf] if c > 0]
    overall_conf = sum(confs) / len(confs) if confs else 0.5

    # ── Try LLM augmentation ──────────────────────────────────────────────────
    llm_rationale = _try_llm_rationale(state, summary_parts, differentials)

    # ── FHIR R4 Output ────────────────────────────────────────────────────────
    from agents.fhir_builder import build_fhir_report
    fhir_report = build_fhir_report(state)

    state["final_report"] = {
        "patient_id": state.get("patient_id", "UNKNOWN"),
        "summary": " ".join(summary_parts),
        "differential_diagnosis": differentials,
        "recommendations": recommendations,
        "overall_confidence": round(overall_conf, 4),
        "imaging_result": imaging,
        "lab_result": lab,
        "nlp_result": nlp,
        "evidence_citations": evidence,
        "llm_rationale": llm_rationale,
        "fhir_report": fhir_report,
    }
    logger.info("[Supervisor] Final report assembled.")
    return state


def _generate_differentials(xray_pred, lab_pred, nlp) -> List[Dict]:
    """Rule-based differential diagnosis generation."""
    differentials = []
    symptoms = nlp.get("symptoms", [])
    conditions = nlp.get("conditions", [])

    if xray_pred == "PNEUMONIA":
        differentials += [
            {"diagnosis": "Community-Acquired Pneumonia", "likelihood": "HIGH",
             "basis": "CXR consolidation pattern + imaging model prediction"},
            {"diagnosis": "Viral Pneumonitis", "likelihood": "MODERATE",
             "basis": "CXR infiltrates; viral etiology possible"},
            {"diagnosis": "Pulmonary Edema", "likelihood": "LOW",
             "basis": "Bilateral opacity pattern — cardiac cause to rule out"},
        ]
    elif xray_pred == "NORMAL":
        differentials += [
            {"diagnosis": "No significant pulmonary pathology", "likelihood": "HIGH",
             "basis": "Normal CXR findings"},
        ]

    if lab_pred == "HIGH RISK":
        differentials += [
            {"diagnosis": "Coronary Artery Disease", "likelihood": "HIGH",
             "basis": "Elevated cardiac risk score, lab profile"},
            {"diagnosis": "Hypertensive Heart Disease", "likelihood": "MODERATE",
             "basis": "Risk factors consistent with hypertensive etiology"},
        ]

    if "chest pain" in symptoms and "shortness of breath" in symptoms:
        differentials.append({"diagnosis": "Acute Coronary Syndrome", "likelihood": "MODERATE",
                               "basis": "Classic ACS symptom constellation"})

    return differentials if differentials else [{"diagnosis": "Insufficient data", "likelihood": "N/A", "basis": "No imaging or lab results available"}]


def _generate_recommendations(xray_pred, lab_pred, nlp) -> List[str]:
    """Generate clinical recommendations based on agent outputs."""
    recs = []
    symptoms = nlp.get("symptoms", [])

    if xray_pred == "PNEUMONIA":
        recs += [
            "Order sputum culture and sensitivity before initiating antibiotics",
            "Initiate empirical antibiotic therapy (amoxicillin 1g TID or doxycycline 100mg BID for outpatient CAP)",
            "Monitor oxygen saturation; supplemental O2 if SpO2 < 94%",
            "CURB-65 scoring to determine inpatient vs outpatient management",
            "Follow-up CXR in 6-8 weeks to confirm resolution",
        ]
    if xray_pred == "NORMAL":
        recs.append("CXR normal — if symptoms persist, consider HRCT chest for occult pathology")

    if lab_pred == "HIGH RISK":
        recs += [
            "Urgent cardiology referral for further evaluation",
            "Fasting lipid profile, HbA1c, renal function panel",
            "12-lead ECG and exercise stress test (or pharmacological stress imaging)",
            "Initiate high-intensity statin therapy (atorvastatin 40-80mg)",
            "Lifestyle counselling: DASH diet, aerobic exercise ≥150 min/week, smoking cessation",
            "Aspirin 75-100mg/day if 10-year CVD risk >10% and no bleeding contraindication",
        ]

    if "fever" in symptoms:
        recs.append("Blood cultures × 2, CBC, CRP, procalcitonin to assess infectious aetiology")
    if "syncope" in symptoms:
        recs.append("Cardiac monitor (telemetry), orthostatic vitals, echocardiography")

    if not recs:
        recs.append("Comprehensive clinical assessment warranted; correlate with full history and examination")

    return recs


def _try_llm_rationale(state, summary_parts, differentials) -> str:
    """Attempt LLM-generated plain-language rationale; fall back to rule-based text."""
    import os
    provider = os.getenv("LLM_PROVIDER", "groq")
    llm = get_llm(provider)

    if llm is None:
        return _rule_based_rationale(summary_parts, differentials)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        sys_msg = SystemMessage(content=(
            "You are MediTwin AI, a clinical decision support system. "
            "Provide a concise, professional 3-4 sentence clinical rationale "
            "for the findings. Be factual, cite the key evidence, and note "
            "that this is an AI-assisted analysis requiring clinician verification."
        ))
        user_msg = HumanMessage(content=(
            f"Clinical Summary: {' '.join(summary_parts)}\n"
            f"Top differentials: {', '.join([d['diagnosis'] for d in differentials[:3]])}\n"
            "Write the clinical rationale:"
        ))
        response = llm.invoke([sys_msg, user_msg])
        return response.content
    except Exception as e:
        logger.warning(f"LLM rationale failed: {e}")
        return _rule_based_rationale(summary_parts, differentials)


def _rule_based_rationale(summary_parts, differentials) -> str:
    top_dx = differentials[0]["diagnosis"] if differentials else "undetermined pathology"
    return (
        f"Based on multimodal analysis of the available clinical data, the primary working diagnosis "
        f"is {top_dx}. {' '.join(summary_parts[:2])} "
        f"This AI-assisted analysis integrates imaging, laboratory, and clinical note data; "
        f"all findings must be correlated with full clinical examination and verified by a licensed clinician."
    )


# ── Build LangGraph Pipeline ──────────────────────────────────────────────────
def build_pipeline():
    """Construct and return the compiled LangGraph StateGraph."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        logger.warning("LangGraph not installed. Using sequential fallback.")
        return None

    graph = StateGraph(ClinicalState)
    graph.add_node("imaging_agent",   imaging_agent)
    graph.add_node("lab_risk_agent",  lab_risk_agent)
    graph.add_node("nlp_agent",       nlp_agent)
    graph.add_node("rag_agent",       rag_agent)
    graph.add_node("supervisor",      supervisor_agent)

    graph.set_entry_point("imaging_agent")
    graph.add_edge("imaging_agent",  "lab_risk_agent")
    graph.add_edge("lab_risk_agent", "nlp_agent")
    graph.add_edge("nlp_agent",      "rag_agent")
    graph.add_edge("rag_agent",      "supervisor")
    graph.add_edge("supervisor",     END)

    return graph.compile()


def run_pipeline(patient_id: str,
                 xray_bytes: Optional[bytes] = None,
                 lab_values: Optional[Dict] = None,
                 clinical_notes: str = "") -> Dict:
    """
    Main entry point for the multi-agent pipeline.
    Works with or without LangGraph (falls back to sequential).
    """
    initial_state: ClinicalState = {
        "patient_id": patient_id,
        "clinical_notes": clinical_notes,
        "lab_values": lab_values or {},
        "xray_image_bytes": xray_bytes,
        "errors": [],
    }

    pipeline = build_pipeline()
    if pipeline:
        final_state = pipeline.invoke(initial_state)
    else:
        # Sequential fallback
        s = initial_state
        for fn in [imaging_agent, lab_risk_agent, nlp_agent, rag_agent, supervisor_agent]:
            s = fn(s)
        final_state = s

    return final_state.get("final_report", {})
