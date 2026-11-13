"""
Train and evaluate a Linear Regression model for student exam-score prediction.
Run from the project root: python src/train_model.py
"""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "outputs" / "figures"
METRICS_PATH = ROOT / "outputs" / "metrics.txt"

FEATURES = [
    "study_hours", "attendance", "previous_score",
    "sleep_hours", "assignments_completed"
]
TARGET = "exam_score"


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = (
        "STUDENT PERFORMANCE PREDICTION - MODEL RESULTS\n"
        "================================================\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows:  {len(X_test)}\n\n"
        f"MAE:  {mae:.2f}\n"
        f"MSE:  {mse:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"R2:   {r2:.3f}\n"
    )
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(metrics, encoding="utf-8")
    print(metrics)

    joblib.dump({"model": model, "features": FEATURES}, MODEL_DIR / "student_score_model.pkl")

    # Actual vs predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.plot([0, 100], [0, 100], linestyle="--")
    plt.xlabel("Actual exam score")
    plt.ylabel("Predicted exam score")
    plt.title("Actual vs Predicted Student Exam Scores")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Residual distribution
    residuals = y_test - predictions
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.xlabel("Prediction error (actual - predicted)")
    plt.ylabel("Number of students")
    plt.title("Distribution of Prediction Errors")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "residuals.png", dpi=150)
    plt.close()

    # Feature coefficients
    coefficients = model.named_steps["regressor"].coef_
    importance = pd.Series(coefficients, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh")
    plt.xlabel("Linear regression coefficient")
    plt.title("Model Feature Coefficients")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_coefficients.png", dpi=150)
    plt.close()

    print(f"Saved trained model to: {MODEL_DIR / 'student_score_model.pkl'}")
    print(f"Saved charts to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
