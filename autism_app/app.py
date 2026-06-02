import io
import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, send_file, jsonify, abort,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.lib import colors

# ---------------------------------------------------------------------------
# Config / model loading
# ---------------------------------------------------------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "autism_model.pkl")
DATASET_PATH = os.path.join(APP_ROOT, "dataset.csv")
HISTORY_PATH = os.path.join(APP_ROOT, "predictions_log.json")

if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        "❌ autism_model.pkl not found. Run `python model_training.py` first."
    )

bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
SCALER = bundle["scaler"]
FEATURES = bundle["features"]
MODEL_NAME = bundle["model_name"]
METRICS = bundle["metrics"]

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_int(value, field, lo=None, hi=None):
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a number.")
    if lo is not None and v < lo:
        raise ValueError(f"'{field}' must be ≥ {lo}.")
    if hi is not None and v > hi:
        raise ValueError(f"'{field}' must be ≤ {hi}.")
    return v


def parse_form(form) -> dict:
    """Validate + normalise the input form into model features."""
    mapping_yes_no = {"yes": 1, "no": 0}
    gender_map = {"male": 1, "female": 0}
    eye_map = {"poor": 1, "normal": 0}

    def yn(field):
        v = (form.get(field) or "").strip().lower()
        if v not in mapping_yes_no:
            raise ValueError(f"'{field}' must be Yes or No.")
        return mapping_yes_no[v]

    data = {
        "age": _to_int(form.get("age"), "Age", 1, 100),
        "gender": gender_map.get((form.get("gender") or "").lower()),
        "family_history": yn("family_history"),
        "speech_delay": yn("speech_delay"),
        "social_difficulty": yn("social_difficulty"),
        "repetitive_behavior": yn("repetitive_behavior"),
        "eye_contact": eye_map.get((form.get("eye_contact") or "").lower()),
        "learning_difficulty": yn("learning_difficulty"),
        "anxiety_level": _to_int(form.get("anxiety_level"),
                                 "Anxiety level", 0, 10),
        "communication_score": _to_int(form.get("communication_score"),
                                       "Communication score", 0, 10),
    }
    if data["gender"] is None:
        raise ValueError("'Gender' must be Male or Female.")
    if data["eye_contact"] is None:
        raise ValueError("'Eye contact' must be Normal or Poor.")
    return data


def predict(features: dict):
    X = pd.DataFrame([[features[f] for f in FEATURES]], columns=FEATURES)
    Xs = SCALER.transform(X)
    pred = int(MODEL.predict(Xs)[0])
    if hasattr(MODEL, "predict_proba"):
        proba = float(MODEL.predict_proba(Xs)[0][pred])
    else:
        proba = 1.0
    return pred, proba


def recommendations(pred: int, features: dict) -> list:
    if pred == 1:
        recs = [
            "Consult a licensed paediatrician or developmental psychologist for a full clinical assessment (e.g. ADOS-2 / ADI-R).",
            "Begin early-intervention therapies — speech, occupational, and behavioural (ABA) — they greatly improve outcomes.",
            "Create a structured daily routine with clear visual schedules.",
            "Join a local or online ASD support community for caregivers.",
        ]
        if features["speech_delay"]:
            recs.append("Prioritise speech-language therapy sessions weekly.")
        if features["anxiety_level"] >= 6:
            recs.append("Discuss anxiety-management strategies (CBT, mindfulness) with a clinician.")
    else:
        recs = [
            "No strong indicators detected — continue monitoring developmental milestones.",
            "Maintain regular paediatric check-ups.",
            "Encourage social play, reading and communication-rich activities.",
            "Re-screen if new behavioural concerns emerge.",
        ]
    recs.append("⚠️ This tool is informational only — it is NOT a medical diagnosis.")
    return recs


def log_prediction(pred: int, proba: float):
    entry = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "prediction": pred,
        "confidence": round(proba * 100, 2),
    }
    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH) as fh:
                history = json.load(fh)
        except Exception:
            history = []
    history.append(entry)
    history = history[-500:]  # cap
    with open(HISTORY_PATH, "w") as fh:
        json.dump(history, fh)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", model_name=MODEL_NAME)


@app.route("/about")
def about():
    df = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else None
    sample_count = int(len(df)) if df is not None else 0
    pos_rate = float(df["label"].mean()) if df is not None else 0.0
    return render_template(
        "about.html",
        model_name=MODEL_NAME,
        metrics=METRICS,
        sample_count=sample_count,
        pos_rate=round(pos_rate * 100, 2),
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        features = parse_form(request.form)
    except ValueError as e:
        return render_template("index.html",
                               model_name=MODEL_NAME, error=str(e)), 400

    pred, proba = predict(features)
    log_prediction(pred, proba)

    label = "Likely Autism Spectrum Disorder (ASD)" if pred == 1 \
        else "No Significant ASD Indicators"
    return render_template(
        "result.html",
        prediction=pred,
        label=label,
        confidence=round(proba * 100, 2),
        features=features,
        recommendations=recommendations(pred, features),
        model_name=MODEL_NAME,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.route("/report.pdf", methods=["POST"])
def report_pdf():
    try:
        features = parse_form(request.form)
    except ValueError:
        abort(400)

    pred, proba = predict(features)
    label = "Likely ASD" if pred == 1 else "No Significant ASD Indicators"

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Autism Screening Report")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Autism Screening Report</b>", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"]))
    story.append(Paragraph(f"Model used: {MODEL_NAME}", styles["Normal"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph(f"<b>Result:</b> {label}", styles["Heading2"]))
    story.append(Paragraph(f"<b>Confidence:</b> {proba*100:.2f}%",
                           styles["Heading3"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Inputs</b>", styles["Heading3"]))
    rows = [["Feature", "Value"]] + [[k, str(v)] for k, v in features.items()]
    t = Table(rows, hAlign="LEFT", colWidths=[200, 200])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.white]),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Recommendations</b>", styles["Heading3"]))
    for r in recommendations(pred, features):
        story.append(Paragraph(f"• {r}", styles["Normal"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "<i>Disclaimer: This report is generated by a machine-learning "
        "screening tool and is NOT a medical diagnosis. Please consult a "
        "qualified clinician.</i>", styles["Italic"]))

    doc.build(story)
    buf.seek(0)
    return send_file(
        buf, mimetype="application/pdf",
        as_attachment=True, download_name="autism_report.pdf",
    )


@app.route("/api/stats")
def api_stats():
    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH) as fh:
                history = json.load(fh)
        except Exception:
            history = []
    total = len(history)
    asd = sum(1 for h in history if h["prediction"] == 1)
    return jsonify({
        "total_predictions": total,
        "asd_predictions": asd,
        "non_asd_predictions": total - asd,
        "model_name": MODEL_NAME,
        "metrics": METRICS,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
