
import streamlit as st
import pandas as pd, numpy as np, joblib, json, yaml
from pathlib import Path
from typing import Dict, Any
from src.risk import RiskThresholds, map_risk

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

def build_pdf_summary(inputs: dict, prob: float, risk: str, narrative: str) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    def text(x, y, s, size=11, bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y, s)

    margin = 2*cm
    y = H - margin
    text(margin, y, "Harmonizing Health and Machine — Mastitis Summary", 14, True); y -= 20
    text(margin, y, f"Risk: {risk}", 12, True); y -= 16
    text(margin, y, f"Probability: {prob:.3f}", 12); y -= 20

    text(margin, y, "Inputs:", 12, True); y -= 14
    for k in ["temperature","milk_visibility","hardness","pain","breed","months_after_giving_birth","previous_mastits_status","cow_id","day"]:
        if k in inputs and inputs[k] not in (None, ""):
            line = f" - {k.replace('_',' ').title()}: {inputs[k]}"
            text(margin, y, line[:95]); y -= 14
            if y < margin+120:
                c.showPage(); y = H - margin

    y -= 6
    text(margin, y, "Interpretation:", 12, True); y -= 14

    # wrap narrative
    import textwrap
    for line in textwrap.wrap(narrative, width=95):
        text(margin, y, line); y -= 14
        if y < margin+40:
            c.showPage(); y = H - margin

    c.showPage()
    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf


st.set_page_config(page_title="Harmonizing Health and Machine — Mastitis", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CONFIG_PATH = ROOT / "config" / "config.yaml"

FRIENDLY_LABELS = {
    "temperature": "Body temperature (°C)",
    "milk_visibility": "Milk appearance",
    "hardness": "Udder hardness (0–2)",
    "pain": "Udder pain (0–2)",
    "breed": "Breed",
    "months_after_giving_birth": "Months after calving",
    "previous_mastits_status": "Previous mastitis (0/1)",
    "cow_id": "Cow ID",
    "day": "Day index",
}
HELP_TEXT = {
    "temperature": "Rectal temperature in °C. Fever ≥ 39.5 °C is a common mastitis sign.",
    "milk_visibility": "Visual check: clots, watery, or blood-tinged milk.",
    "hardness": "0 = normal, 1 = firm, 2 = very hard.",
    "pain": "0 = none, 1 = mild, 2 = severe pain on palpation.",
    "breed": "Optional. If unknown, leave blank.",
    "months_after_giving_birth": "Number of months since last calving.",
    "previous_mastits_status": "Has this cow had mastitis before? 0 = No, 1 = Yes.",
    "cow_id": "Your own identifier for the cow (tag/ear number).",
    "day": "Day index for time series (optional).",
}

BASIC_FIELDS = ["temperature", "milk_visibility"]
CLINICAL_OPTIONAL = ["hardness", "pain"]
CONTEXT_FIELDS = ["breed","months_after_giving_birth","previous_mastits_status","cow_id","day"]

def label_for(c): return FRIENDLY_LABELS.get(c, c.replace("_"," ").title())
def help_for(c): return HELP_TEXT.get(c,"")

# ----- Explanation helpers (shown only after Predict) -----
def _fmt_temp_c(val):
    try:
        v = float(val); return f"{v:.1f} °C"
    except Exception:
        return str(val)

def interpret_inputs(inputs: dict) -> list[str]:
    bullets = []
    temp = inputs.get("temperature", None)
    if temp is not None:
        try:
            t = float(temp)
            if t < 38.0: tag = "This is **below normal** (cow baseline ≈ 38.5–39.5 °C)."
            elif t <= 39.5: tag = "This is **normal** (≈ 38.5–39.5 °C)."
            else: tag = "This is **elevated**; fever can indicate mastitis."
        except Exception:
            tag = ""
        bullets.append(f"**Body temperature:** { _fmt_temp_c(temp) } — {tag}")
    mv = inputs.get("milk_visibility", "")
    if mv:
        flag = "🚩 " if str(mv).strip().lower()=="abnormal" else ""
        bullets.append(f"**Milk appearance:** {mv} {flag}")
    for name, title in [("hardness","Udder hardness (0–2)"),
                        ("pain","Udder pain (0–2)"),
                        ("breed","Breed"),
                        ("months_after_giving_birth","Months after calving"),
                        ("previous_mastits_status","Previous mastitis (0/1)"),
                        ("cow_id","Cow ID"),
                        ("day","Day index")]:
        if name in inputs and inputs.get(name) not in (None, ""):
            bullets.append(f"**{title}:** {inputs[name]}")
    return bullets

def interpret_decision(inputs: dict, prob: float, risk: str) -> str:
    cues = []
    if str(inputs.get("milk_visibility","")).lower()=="abnormal":
        cues.append("abnormal milk")
    try:
        if float(inputs.get("hardness",0)) >= 1: cues.append("udder hardness")
    except Exception: pass
    try:
        if float(inputs.get("pain",0)) >= 1: cues.append("udder pain")
    except Exception: pass
    try:
        t = float(inputs.get("temperature",0))
        if t >= 39.5: cues.append("fever")
        elif t < 38.0: cues.append("below-normal temperature (possible input error)")
    except Exception: pass

    if risk in ("URGENT","HIGH"):
        body = (("The combination of " + ", ".join(cues) + " and ") if cues else "") + f"the model score ({prob:.2f}) suggests **mastitis is likely**."
        advice = "→ **Action:** Isolate the cow, perform CMT/strip cup test, consult a vet early."
    elif risk == "MEDIUM":
        body = f"The model score ({prob:.2f}) suggests **possible mastitis**. Re-check tomorrow or add more measurements."
        advice = "→ **Action:** Monitor closely; consider CMT if milk remains abnormal."
    else:
        body = f"The model score ({prob:.2f}) suggests **low probability** of mastitis with current inputs."
        advice = "→ **Action:** Continue routine checks; if signs change, re-evaluate."
    return body + " " + advice

def risk_banner(risk: str) -> str:
    colors = {
        "LOW":    ("#0f5132", "#d1e7dd"),   # green text on light green
        "MEDIUM": ("#664d03", "#fff3cd"),   # dark yellow on light yellow
        "HIGH":   ("#842029", "#f8d7da"),   # dark red on light pink
        "URGENT": ("#ffffff", "#dc3545"),   # white on red
    }
    fg, bg = colors.get(risk, ("#0f5132", "#d1e7dd"))
    return f'''
    <div style="padding:12px 16px;border-radius:8px;background:{bg};
                display:flex;align-items:center;gap:12px;margin:8px 0;">
        <div style="font-size:20px;">{'🟢' if risk=='LOW' else '🟡' if risk=='MEDIUM' else '🟠' if risk=='HIGH' else '🔴'}</div>
        <div style="color:{fg};font-weight:700;letter-spacing:0.5px;">
            RISK: {risk}
        </div>
    </div>
    '''

@st.cache_resource
def load_config_and_artifacts():
    cfg = yaml.safe_load(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    thr_cfg = cfg.get("risk_thresholds", {"urgent":0.80,"high":0.55,"low":0.35})
    thr = RiskThresholds(float(thr_cfg["urgent"]), float(thr_cfg["high"]), float(thr_cfg["low"]))
    best_model_path = MODELS_DIR / "best_model.joblib"
    schema_path = MODELS_DIR / "schema.json"
    model = joblib.load(best_model_path) if best_model_path.exists() else None
    schema = json.loads(schema_path.read_text()) if schema_path.exists() else {"numeric": [], "categorical": [], "target": "class1"}
    return cfg, thr, model, schema

def build_inputs(schema):
    numeric = set(schema.get("numeric", []))
    categorical = set(schema.get("categorical", []))
    vals = {}
    with st.form("mastitis_form"):
        st.markdown("### 1) Basic checks")
        c1, c2 = st.columns(2)
        if "temperature" in numeric:
            vals["temperature"] = c1.number_input(label_for("temperature"), help=help_for("temperature"), value=0.0)
        if "milk_visibility" in categorical:
            vals["milk_visibility"] = c2.selectbox(label_for("milk_visibility"), ["Normal","Abnormal"], help=help_for("milk_visibility"))
        with st.expander("2) Clinical observations (optional)"):
            d1, d2 = st.columns(2)
            if "hardness" in numeric:
                vals["hardness"] = d1.number_input(label_for("hardness"), help=help_for("hardness"), value=0.0)
            elif "hardness" in categorical:
                vals["hardness"] = d1.selectbox(label_for("hardness"), [0,1,2], help=help_for("hardness"))
            if "pain" in numeric:
                vals["pain"] = d2.number_input(label_for("pain"), help=help_for("pain"), value=0.0)
            elif "pain" in categorical:
                vals["pain"] = d2.selectbox(label_for("pain"), [0,1,2], help=help_for("pain"))
        with st.expander("3) Context (optional)"):
            e1, e2 = st.columns(2)
            for i, col in enumerate(["breed","months_after_giving_birth","previous_mastits_status","cow_id","day"]):
                dest = e1 if i%2==0 else e2
                if col in numeric:
                    vals[col] = dest.number_input(label_for(col), help=help_for(col), value=0.0)
                elif col in categorical:
                    vals[col] = dest.text_input(label_for(col), help=help_for(col))
                else:
                    vals[col] = dest.text_input(label_for(col), help=help_for(col))
        submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")
    return vals, submitted

def main():
    st.title("🐄 Harmonizing Health and Machine — Mastitis Detection (Farmer Minimal)")
    st.caption("Enter temperature, milk appearance, and optional hardness/pain. Leave rest blank.")
    cfg, thr, model, schema = load_config_and_artifacts()
    tabs = st.tabs(["Predict", "Batch (CSV)", "Model"])

    with tabs[0]:
        inputs, go = build_inputs(schema)
        if go:
            if not schema["numeric"] and not schema["categorical"]:
                st.error("Schema not found. Train once to generate models/schema.json.")
                st.stop()
            X = pd.DataFrame([inputs])
            if model is None:
                st.warning("Model not loaded. Please train first. Showing a demo probability.")
                prob = float(np.clip(np.random.normal(0.5, 0.1), 0, 1))
            else:
                try: prob = float(model.predict_proba(X)[:,1][0])
                except Exception:
                    score = model.decision_function(X)[0]
                    prob = float(np.clip((score + 3)/6.0, 0, 1))
            risk = map_risk(prob, thr)
            # Colored risk banner
            st.markdown(risk_banner(risk), unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Mastitis Probability", f"{prob:.3f}")
            c2.metric("Risk Level", risk)

            st.markdown("#### Situation summary")
            for b in interpret_inputs(inputs):
                st.markdown(f"- {b}")

            st.markdown("#### Interpretation")
            narrative = interpret_decision(inputs, prob, risk)
            st.write(narrative)
            pdf_bytes = build_pdf_summary(inputs, prob, risk, narrative)
            st.download_button("Download PDF summary", data=pdf_bytes, file_name="mastitis_summary.pdf", mime="application/pdf")

            st.markdown("#### Top-3 possible states")
            top3 = [
                {"condition": "Mastitis", "score": prob},
                {"condition": "Healthy", "score": 1 - prob},
                {"condition": "Other (monitor)", "score": 0.25*(1-abs(0.5-prob))}
            ]
            st.table(pd.DataFrame(top3))

    with tabs[1]:
        st.subheader("Batch Inference")
        st.write("Upload a CSV with the same columns used to train (exclude the target).")
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            if model is None:
                st.warning("Model not loaded. Train first to get real predictions. Showing demo scores.")
                probs = np.clip(np.random.normal(0.5, 0.15, size=len(df)), 0, 1)
            else:
                try: probs = model.predict_proba(df)[:,1]
                except Exception:
                    score = model.decision_function(df)
                    mins, maxs = float(np.min(score)), float(np.max(score))
                    probs = (score - mins) / (maxs - mins + 1e-9)
            df_out = df.copy()
            df_out["prob_mastitis"] = probs
            df_out["risk"] = [map_risk(float(p), thr) for p in probs]
            st.dataframe(df_out)

    with tabs[2]:
        st.subheader("Model Info")
        st.write("Schema (numeric vs categorical):")
        st.json(schema)
        lb_path = MODELS_DIR/"leaderboard.json"
        if lb_path.exists():
            st.markdown("**Leaderboard:**")
            st.json(json.loads(lb_path.read_text()))
        else:
            st.info("Leaderboard not found. Train models first.")

if __name__ == "__main__":
    main()
