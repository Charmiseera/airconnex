"""
ml/predict.py
=============
Inference helper — loads saved model + scaler and returns prediction.
Used both for testing and by the FastAPI backend (Phase 4).

Usage (standalone):
    python ml/predict.py
"""

import os, json
import numpy as np
import joblib

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR   = os.path.join(BASE, "ml")
MODEL_F  = os.path.join(ML_DIR, "model.pkl")
SCALER_F = os.path.join(ML_DIR, "scaler.pkl")
FEAT_F   = os.path.join(ML_DIR, "feature_names.json")

CLASS_NAMES  = {0: "Risky", 1: "Tight", 2: "Safe"}
CLASS_COLORS = {"Safe": "green", "Tight": "orange", "Risky": "red"}

# ── Load artefacts once ────────────────────────────────────────────────────────
_model   = None
_scaler  = None
_features= None

def _load():
    global _model, _scaler, _features
    if _model is None:
        _model   = joblib.load(MODEL_F)
        _scaler  = joblib.load(SCALER_F)
        with open(FEAT_F) as f:
            _features = json.load(f)
    return _model, _scaler, _features


def predict(input_dict: dict) -> dict:
    """
    Parameters
    ----------
    input_dict : dict
        Keys must match feature names from feature_names.json.
        Missing numeric keys default to 0.

    Returns
    -------
    dict with keys:
        risk_label     : int   (0=Risky, 1=Tight, 2=Safe)
        risk_category  : str   ("Risky" | "Tight" | "Safe")
        color          : str   ("red" | "orange" | "green")
        probabilities  : dict  {class_name: probability}
        confidence     : float  (max probability)
    """
    model, scaler, features = _load()

    # Build feature vector in correct order
    row = np.array([[float(input_dict.get(f, 0)) for f in features]])

    # Scale
    row_sc = scaler.transform(row)

    # Predict
    label      = int(model.predict(row_sc)[0])
    proba      = model.predict_proba(row_sc)[0]

    category   = CLASS_NAMES[label]
    color      = CLASS_COLORS[category]
    proba_dict = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)}
    confidence = round(float(proba.max()), 4)

    return {
        "risk_label"   : label,
        "risk_category": category,
        "color"        : color,
        "probabilities": proba_dict,
        "confidence"   : confidence,
    }


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Smart Connection Feasibility Predictor — Inference Test ===\n")

    test_cases = [
        {
            "name": "Safe connection (DXB, same terminal, no delay)",
            "input": {
                "connection_time_min" : 120,
                "arrival_delay_min"   : 0,
                "terminal_walk_time_min": 6,
                "terminal_distance_m" : 400,
                "transport_friction"  : 1,
                "is_intl_hub"         : 1,
                "is_peak_hour"        : 0,
                "terminal_change"     : 0,
                "departure_hour"      : 14,
                "day_of_week"         : 2,
                "walking_time_min"    : 10,
                "security_time_min"   : 20,
                "immigration_time_min": 30,
                "congestion_time_min" : 8,
                "baggage_time_min"    : 10,
            }
        },
        {
            "name": "Tight connection (LHR, terminal change, small delay)",
            "input": {
                "connection_time_min" : 60,
                "arrival_delay_min"   : 15,
                "terminal_walk_time_min": 15,
                "terminal_distance_m" : 900,
                "transport_friction"  : 2,
                "is_intl_hub"         : 1,
                "is_peak_hour"        : 1,
                "terminal_change"     : 1,
                "departure_hour"      : 8,
                "day_of_week"         : 0,
                "walking_time_min"    : 25,
                "security_time_min"   : 20,
                "immigration_time_min": 30,
                "congestion_time_min" : 15,
                "baggage_time_min"    : 10,
            }
        },
        {
            "name": "Risky connection (ORD, heavy delay, immigration)",
            "input": {
                "connection_time_min" : 55,
                "arrival_delay_min"   : 35,
                "terminal_walk_time_min": 20,
                "terminal_distance_m" : 1200,
                "transport_friction"  : 3,
                "is_intl_hub"         : 1,
                "is_peak_hour"        : 1,
                "terminal_change"     : 1,
                "departure_hour"      : 17,
                "day_of_week"         : 4,
                "walking_time_min"    : 25,
                "security_time_min"   : 20,
                "immigration_time_min": 30,
                "congestion_time_min" : 15,
                "baggage_time_min"    : 10,
            }
        },
    ]

    for tc in test_cases:
        result = predict(tc["input"])
        print(f"  Scenario : {tc['name']}")
        print(f"  Result   : {result['risk_category']}  (confidence {result['confidence']:.1%})")
        print(f"  Probs    : {result['probabilities']}")
        print()
