"""
Quick test script to debug prediction errors
"""
import os
import sys
import joblib
import pandas as pd

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "autism_model.pkl")

try:
    print("Loading model...")
    bundle = joblib.load(MODEL_PATH)
    MODEL = bundle["model"]
    SCALER = bundle["scaler"]
    FEATURES = bundle["features"]
    MODEL_NAME = bundle["model_name"]
    
    print(f"Model: {MODEL_NAME}")
    print(f"Features: {FEATURES}")
    print()
    
    # Test with valid data
    test_data = {
        "age": 7,
        "gender": 1,
        "family_history": 1,
        "speech_delay": 1,
        "social_difficulty": 1,
        "repetitive_behavior": 1,
        "eye_contact": 1,
        "learning_difficulty": 1,
        "anxiety_level": 5,
        "communication_score": 4,
    }
    
    print(f"Test data: {test_data}")
    print()
    
    # Create DataFrame
    print("Creating DataFrame...")
    X = pd.DataFrame([[test_data[f] for f in FEATURES]], columns=FEATURES)
    print(f"DataFrame shape: {X.shape}")
    print(f"DataFrame:\n{X}")
    print()
    
    # Scale
    print("Scaling...")
    Xs = SCALER.transform(X)
    print(f"Scaled shape: {Xs.shape}")
    print()
    
    # Predict
    print("Predicting...")
    pred = int(MODEL.predict(Xs)[0])
    print(f"Prediction: {pred}")
    
    # Get probability
    print("Getting probability...")
    if hasattr(MODEL, "predict_proba"):
        print(f"predict_proba available: {MODEL.predict_proba(Xs)}")
        proba_array = MODEL.predict_proba(Xs)[0]
        print(f"Probability array: {proba_array}")
        proba = float(proba_array[pred])
        print(f"Probability for pred {pred}: {proba}")
    else:
        proba = 1.0
        print(f"No predict_proba, using default: {proba}")
    
    print()
    print("✅ Prediction successful!")
    print(f"Result: Prediction={pred}, Confidence={proba*100:.2f}%")
    
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

