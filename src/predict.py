"""Predict an exam score for one student using the saved model."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "student_score_model.pkl"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. First run: python src/train_model.py")

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]

    # Example student profile - edit these values to try another prediction.
    student = pd.DataFrame([{
        "study_hours": 5.0,
        "attendance": 85.0,
        "previous_score": 72.0,
        "sleep_hours": 7.0,
        "assignments_completed": 8
    }], columns=features)

    predicted_score = model.predict(student)[0]
    print(f"Predicted exam score: {predicted_score:.2f} / 100")

if __name__ == "__main__":
    main()
