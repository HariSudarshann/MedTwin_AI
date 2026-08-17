"""
MediTwin AI — Streamlit Demo UI (Review 3)
"""
import sys, os, json, io, base64, time
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediTwin AI — Clinical Decision Support",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #091520 100%);
    border-right: 1px solid #1e3a5f;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #112240 0%, #0a192f 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.agent-card {
    background: linear-gradient(135deg, #0d2137 0%, #091828 100%);
    border: 1px solid #1a4a7a;
    border-radius: 12px;
    padding: 18px;
    margin: 10px 0;
    position: relative;
}
.agent-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 8px;
}
.badge-imaging  { background:#1a4a8a; color:#64b5f6; }
.badge-lab      { background:#1a5a3a; color:#66bb6a; }
.badge-nlp      { background:#4a3a1a; color:#ffa726; }
.badge-rag      { background:#3a1a5a; color:#ab47bc; }
.badge-supervisor{ background:#5a1a1a; color:#ef5350; }

/* Prediction boxes */
.pred-normal   { background:linear-gradient(135deg,#0d3325,#0a2a1e); border:1px solid #2e7d52; border-radius:10px; padding:16px; }
.pred-pneumonia{ background:linear-gradient(135deg,#3d1515,#2a0d0d); border:1px solid #c62828; border-radius:10px; padding:16px; }
.pred-high     { background:linear-gradient(135deg,#3d2a00,#2a1c00); border:1px solid #f57c00; border-radius:10px; padding:16px; }
.pred-low      { background:linear-gradient(135deg,#0d3325,#0a2a1e); border:1px solid #388e3c; border-radius:10px; padding:16px; }

/* Confidence bar */
.conf-bar-wrap { background:#1e2a3a; border-radius:8px; height:12px; margin:6px 0; overflow:hidden; }
.conf-bar      { height:100%; border-radius:8px; transition: width 0.6s ease; }

/* Citation card */
.citation-card {
    background:#0d1f35; border:1px solid #1e3a5f;
    border-left: 4px solid #4f86f7;
    border-radius:8px; padding:14px; margin:8px 0;
}

/* FHIR block */
.fhir-block {
    background:#050d18; border:1px solid #0d2137;
    border-radius:8px; padding:16px; font-family:monospace;
    font-size:12px; color:#7ec8e3; max-height:400px; overflow-y:auto;
}

/* Section headers */
.section-header {
    color:#4f86f7; font-size:22px; font-weight:700;
    border-bottom:2px solid #1e3a5f; padding-bottom:8px; margin:20px 0 16px;
}
.sub-header { color:#90caf9; font-size:15px; font-weight:600; margin:12px 0 6px; }

/* Status pill */
.status-ok   { color:#66bb6a; font-weight:600; }
.status-warn { color:#ffa726; font-weight:600; }
.status-err  { color:#ef5350; font-weight:600; }

/* Scrollbar */
::-webkit-scrollbar { width:6px; } ::-webkit-scrollbar-track { background:#0a0e1a; }
::-webkit-scrollbar-thumb { background:#1e3a5f; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Helper utilities ──────────────────────────────────────────────────────────
def b64_to_img(b64_str):
    return io.BytesIO(base64.b64decode(b64_str))

def conf_bar(value, color="#4f86f7"):
    pct = int(value * 100)
    return f"""
    <div class='conf-bar-wrap'>
      <div class='conf-bar' style='width:{pct}%; background:{color};'></div>
    </div>
    <small style='color:#90caf9;'>{pct}% confidence</small>
    """

def pred_class(pred):
    p = (pred or "").upper()
    if "PNEUMONIA" in p: return "pred-pneumonia"
    if "HIGH" in p:      return "pred-high"
    if "NORMAL" in p:    return "pred-normal"
    return "pred-low"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 MediTwin AI")
    st.markdown("<small style='color:#607d8b;'>Review 3 — Capstone Demo</small>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📋 Patient Information")
    patient_id = st.text_input("Patient ID", value="PT-2026-001", key="pid")
    patient_name = st.text_input("Patient Name (optional)", value="", key="pname")
    patient_age = st.number_input("Age", 18, 100, 55, key="page")
    patient_sex = st.selectbox("Sex", ["Male", "Female"], key="psex")

    st.divider()

    # Model status
    st.markdown("### 🤖 Model Status")
    xray_ready = (ROOT / "models" / "resnet18_xray.pt").exists()
    tab_ready  = (ROOT / "models" / "heart_risk_rf.joblib").exists()
    st.markdown(f"🫁 Imaging Model: {'<span class=\"status-ok\">✓ Ready</span>' if xray_ready else '<span class=\"status-warn\">⚠ Not trained</span>'}", unsafe_allow_html=True)
    st.markdown(f"🫀 Tabular Model: {'<span class=\"status-ok\">✓ Ready</span>' if tab_ready else '<span class=\"status-warn\">⚠ Not trained</span>'}", unsafe_allow_html=True)

    st.divider()
    if st.button("🚀 Train Models Now", use_container_width=True, type="secondary"):
        with st.spinner("Training models (this takes ~5-10 min on CPU)..."):
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "train_models.py")],
                capture_output=True, text=True, cwd=str(ROOT)
            )
            if result.returncode == 0:
                st.success("✅ Models trained successfully!")
            else:
                st.error(f"Training error: {result.stderr[-500:]}")

    st.divider()
    st.markdown("<small style='color:#546e7a;'>© 2026 MediTwin AI — B.Tech Capstone<br>AI & Data Science Dept.</small>", unsafe_allow_html=True)

# ── Main Layout ────────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#4f86f7; margin-bottom:4px;'>🧬 MediTwin AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#607d8b; font-size:15px;'>Multimodal Multi-Agent Clinical Decision Support System — Review 3</p>", unsafe_allow_html=True)

tabs = st.tabs(["📥 Patient Intake", "🫁 Imaging Analysis", "🫀 Lab / Risk", "🔬 NLP Insights", "📚 Evidence (RAG)", "📊 Clinical Report", "🏥 FHIR R4 Export"])

# ── TAB 1: Patient Intake ─────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("<div class='section-header'>📥 Patient Data Input</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div class='sub-header'>🫁 Chest X-Ray Upload</div>", unsafe_allow_html=True)
        xray_file = st.file_uploader("Upload Chest X-Ray (JPG/PNG)", type=["jpg", "jpeg", "png"], key="xray_up")
        if xray_file:
            st.image(xray_file, caption="Uploaded X-Ray", use_container_width=True)

    with col2:
        st.markdown("<div class='sub-header'>📝 Clinical Notes / History</div>", unsafe_allow_html=True)
        clinical_notes = st.text_area(
            "Paste clinical notes, discharge summary, or symptoms",
            height=200,
            value="Patient presents with fever (38.8°C), productive cough, and shortness of breath for 3 days. "
                  "BP: 145/92 mmHg, HR: 98 bpm, SpO2: 94%. "
                  "History of hypertension and type 2 diabetes. "
                  "On metformin and amlodipine. Chest pain reported on exertion.",
            key="notes"
        )

    st.markdown("<div class='sub-header'>🧪 Laboratory Values (Cardiac Risk Panel)</div>", unsafe_allow_html=True)
    st.info("Enter lab values for cardiac risk assessment. Leave defaults if not applicable.")

    cols = st.columns(4)
    lab_fields = {
        "age":      ("Age (yrs)",          patient_age, 18, 100, 1),
        "sex":      ("Sex (1=M, 0=F)",     1 if patient_sex=="Male" else 0, 0, 1, 1),
        "cp":       ("Chest Pain Type 0-3",2, 0, 3, 1),
        "trestbps": ("Resting BP (mmHg)", 140, 80, 220, 1),
        "chol":     ("Cholesterol mg/dL", 220, 100, 500, 1),
        "fbs":      ("Fasting BS >120 (1/0)",0, 0, 1, 1),
        "restecg":  ("Resting ECG 0-2",   1, 0, 2, 1),
        "thalach":  ("Max Heart Rate",    148, 60, 220, 1),
        "exang":    ("Exertional Angina",  0, 0, 1, 1),
        "oldpeak":  ("ST Depression",      1.2, 0.0, 8.0, 0.1),
        "slope":    ("ST Slope 0-2",       1, 0, 2, 1),
        "ca":       ("Vessels (0-4)",      0, 0, 4, 1),
        "thal":     ("Thalassemia 0-3",    2, 0, 3, 1),
    }
    lab_values = {}
    for i, (key, (label, default, mn, mx, step)) in enumerate(lab_fields.items()):
        with cols[i % 4]:
            if isinstance(default, float) or isinstance(step, float):
                lab_values[key] = st.number_input(label, float(mn), float(mx), float(default), step=float(step), key=f"lab_{key}")
            else:
                lab_values[key] = st.number_input(label, int(mn), int(mx), int(default), step=int(step), key=f"lab_{key}")

    st.divider()
    run_btn = st.button("🚀 Run MediTwin AI Analysis", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("🧬 Running multi-agent clinical analysis..."):
            progress = st.progress(0, text="Initialising agents...")
            try:
                from agents.pipeline import run_pipeline
                xray_bytes = xray_file.read() if xray_file else None
                if xray_file: xray_file.seek(0)

                progress.progress(20, text="🫁 Imaging Agent running...")
                time.sleep(0.3)
                progress.progress(40, text="🫀 Lab/Risk Agent running...")
                time.sleep(0.3)
                progress.progress(60, text="🔬 NLP Agent parsing notes...")
                time.sleep(0.3)
                progress.progress(75, text="📚 RAG Evidence retrieval...")
                time.sleep(0.3)
                progress.progress(90, text="🧠 Supervisor synthesising report...")

                report = run_pipeline(
                    patient_id=patient_id,
                    xray_bytes=xray_bytes,
                    lab_values=lab_values,
                    clinical_notes=clinical_notes,
                )
                st.session_state["report"] = report
                st.session_state["xray_file_bytes"] = xray_bytes
                progress.progress(100, text="✅ Analysis complete!")
                st.success("✅ Analysis complete! Navigate to the tabs above to view results.")
            except Exception as e:
                st.error(f"Analysis error: {e}")
                import traceback; st.code(traceback.format_exc())


# ── Helper: get report ────────────────────────────────────────────────────────
def get_report():
    return st.session_state.get("report", {})

# ── TAB 2: Imaging ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("<div class='section-header'>🫁 Imaging Agent — Chest X-Ray Analysis</div>", unsafe_allow_html=True)
    report = get_report()
    imaging = report.get("imaging_result", {})

    if not report:
        st.info("Run analysis from the Patient Intake tab to see results here.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Original X-Ray**")
            xb = st.session_state.get("xray_file_bytes")
            if xb:
                from PIL import Image
                img = Image.open(io.BytesIO(xb))
                st.image(img, use_container_width=True)
            else:
                st.info("No X-ray image uploaded")

        with col2:
            st.markdown("**Grad-CAM Heatmap**")
            gcam = imaging.get("gradcam_b64")
            if gcam:
                st.image(b64_to_img(gcam), caption="Grad-CAM — highlighted regions influenced prediction", use_container_width=True)
            else:
                st.info("Grad-CAM available after uploading an X-ray")

        pred = imaging.get("prediction", "N/A")
        conf = imaging.get("confidence", 0)
        css_class = pred_class(pred)
        icon = "🔴" if "PNEUMONIA" in str(pred).upper() else "🟢"
        st.markdown(f"<div class='{css_class}'><h3>{icon} {pred}</h3>{conf_bar(conf, '#ef5350' if 'PNEUMONIA' in str(pred).upper() else '#66bb6a')}</div>", unsafe_allow_html=True)

        probs = imaging.get("probabilities", {})
        if probs:
            st.markdown("**Class Probabilities**")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#0a192f")
            labels, vals = list(probs.keys()), list(probs.values())
            colors = ["#66bb6a" if "NORMAL" in l else "#ef5350" for l in labels]
            bars = ax.barh(labels, vals, color=colors, height=0.4)
            ax.set_xlim(0, 1); ax.set_facecolor("#0a192f")
            ax.tick_params(colors="white"); ax.spines[:].set_color("#1e3a5f")
            for bar, v in zip(bars, vals):
                ax.text(v + 0.01, bar.get_y() + bar.get_height()/2, f"{v:.1%}", va="center", color="white", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        if imaging.get("note"):
            st.warning(imaging["note"])

# ── TAB 3: Lab / Risk ─────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-header'>🫀 Lab/Risk Agent — Cardiac Risk Assessment</div>", unsafe_allow_html=True)
    report = get_report()
    lab = report.get("lab_result", {})

    if not report:
        st.info("Run analysis from the Patient Intake tab.")
    else:
        pred = lab.get("prediction", "N/A")
        conf = lab.get("confidence", 0)
        css_class = pred_class(pred)
        icon = "🔴" if "HIGH" in str(pred).upper() else "🟢"
        st.markdown(f"<div class='{css_class}'><h3>{icon} {pred}</h3>{conf_bar(conf, '#ffa726' if 'HIGH' in str(pred).upper() else '#66bb6a')}</div>", unsafe_allow_html=True)
        st.markdown("")

        col1, col2 = st.columns([1, 1])
        with col1:
            shap_b64 = lab.get("shap_waterfall_b64")
            if shap_b64:
                st.markdown("**SHAP Waterfall — Feature Contributions**")
                st.image(b64_to_img(shap_b64), use_container_width=True)
            else:
                st.info("SHAP plot available after training the model.")

        with col2:
            top_factors = lab.get("top_risk_factors", [])
            if top_factors:
                st.markdown("**Top Risk Factors (SHAP)**")
                import matplotlib.pyplot as plt
                feats = [f for f, v in top_factors]
                vals  = [v for f, v in top_factors]
                colors = ["#ef5350" if v > 0 else "#66bb6a" for v in vals]
                fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0a192f")
                ax.barh(feats, vals, color=colors, height=0.5)
                ax.axvline(0, color="white", lw=0.8, ls="--")
                ax.set_facecolor("#0a192f"); ax.tick_params(colors="white")
                ax.spines[:].set_color("#1e3a5f")
                ax.set_title("SHAP Feature Impact", color="#90caf9")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

        if lab.get("note"):
            st.warning(lab["note"])

# ── TAB 4: NLP ────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("<div class='section-header'>🔬 NLP Agent — Clinical Notes Analysis</div>", unsafe_allow_html=True)
    report = get_report()
    nlp = report.get("nlp_result", {})

    if not report:
        st.info("Run analysis from the Patient Intake tab.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🤒 Symptoms Detected**")
            syms = nlp.get("symptoms", [])
            if syms:
                for s in syms: st.markdown(f"• {s.title()}")
            else: st.caption("None detected")

        with c2:
            st.markdown("**🏥 Known Conditions**")
            conds = nlp.get("conditions", [])
            if conds:
                for c in conds: st.markdown(f"• {c.title()}")
            else: st.caption("None detected")

        with c3:
            st.markdown("**💊 Medications**")
            meds = nlp.get("medications", [])
            if meds:
                for m in meds: st.markdown(f"• {m.title()}")
            else: st.caption("None detected")

        vitals = nlp.get("vitals", {})
        if vitals:
            st.markdown("**📊 Extracted Vitals**")
            vc = st.columns(len(vitals))
            for i, (k, v) in enumerate(vitals.items()):
                with vc[i]: st.metric(k.replace("_", " ").title(), v)

# ── TAB 5: RAG Evidence ────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("<div class='section-header'>📚 Evidence Retrieval Agent — Clinical Guidelines</div>", unsafe_allow_html=True)
    report = get_report()
    evidence = report.get("evidence_citations", [])

    if not report:
        st.info("Run analysis from the Patient Intake tab.")
    elif not evidence:
        st.warning("No evidence retrieved. Check ChromaDB setup.")
    else:
        st.markdown(f"**{len(evidence)} Evidence Citations Retrieved**")
        for ev in evidence:
            score_color = "#66bb6a" if ev.get("relevance_score", 0) > 0.5 else "#ffa726"
            st.markdown(f"""
            <div class='citation-card'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <strong style='color:#4f86f7;'>[{ev['rank']}] {ev['title']}</strong>
                    <span style='color:{score_color}; font-size:12px; font-weight:600;'>
                        Relevance: {ev.get('relevance_score', 0):.2f}
                    </span>
                </div>
                <div style='color:#607d8b; font-size:12px; margin:4px 0;'>📖 {ev['source']}</div>
                <div style='color:#b0bec5; font-size:13px; margin-top:8px;'>{ev['snippet']}</div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 6: Clinical Report ─────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("<div class='section-header'>📊 Clinical Report — Supervisor Agent</div>", unsafe_allow_html=True)
    report = get_report()

    if not report:
        st.info("Run analysis from the Patient Intake tab.")
    else:
        # Summary
        st.markdown("**🩺 Clinical Summary**")
        st.markdown(f"<div class='agent-card'>{report.get('summary', 'N/A')}</div>", unsafe_allow_html=True)

        # LLM Rationale
        rationale = report.get("llm_rationale", "")
        if rationale:
            st.markdown("**🧠 AI Clinical Rationale**")
            st.markdown(f"<div class='agent-card' style='border-color:#7e57c2;'>{rationale}</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔬 Differential Diagnosis**")
            for d in report.get("differential_diagnosis", []):
                lh = d.get("likelihood", "")
                color = "#ef5350" if lh == "HIGH" else "#ffa726" if lh == "MODERATE" else "#66bb6a"
                st.markdown(f"<div class='agent-card' style='border-left:4px solid {color};'><b style='color:{color};'>{lh}</b> — {d['diagnosis']}<br><small style='color:#607d8b;'>{d.get('basis','')}</small></div>", unsafe_allow_html=True)

        with col2:
            st.markdown("**📋 Treatment Recommendations**")
            for i, r in enumerate(report.get("recommendations", []), 1):
                st.markdown(f"**{i}.** {r}")

        # Overall confidence
        oc = report.get("overall_confidence", 0)
        st.markdown(f"**Overall Confidence:** {oc:.1%}")
        st.progress(oc)

        # PDF export
        if st.button("📄 Download PDF Report"):
            try:
                from scripts.generate_report import generate_pdf
                pdf_path = generate_pdf(report, patient_id)
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇ Download PDF", f, file_name=f"MediTwin_{patient_id}.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF generation error: {e}")

# ── TAB 7: FHIR ───────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("<div class='section-header'>🏥 FHIR R4 Export</div>", unsafe_allow_html=True)
    report = get_report()

    if not report:
        st.info("Run analysis from the Patient Intake tab.")
    else:
        fhir = report.get("fhir_report", {})
        st.markdown("**FHIR R4 Bundle (DiagnosticReport + Observations)**")
        st.caption("⚠ AI-generated output. Requires clinician validation before EHR submission.")

        fhir_json = json.dumps(fhir, indent=2, default=str)
        st.markdown(f"<div class='fhir-block'><pre>{fhir_json[:3000]}{'...' if len(fhir_json)>3000 else ''}</pre></div>", unsafe_allow_html=True)

        st.download_button(
            "⬇ Download FHIR R4 JSON",
            fhir_json.encode(),
            file_name=f"MediTwin_FHIR_{patient_id}.json",
            mime="application/fhir+json"
        )

        # Show resource summary
        if fhir.get("entry"):
            st.markdown("**Bundle Contents**")
            for e in fhir["entry"]:
                r = e.get("resource", {})
                st.markdown(f"• `{r.get('resourceType','')}` — {r.get('id','')[:16]}...")
