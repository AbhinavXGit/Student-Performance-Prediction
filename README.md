# Student Performance Prediction Using Machine Learning

A beginner-friendly regression project that predicts a student's exam score from study habits, attendance, and previous academic performance.

## Project overview

This project uses **Linear Regression** to estimate exam scores from five input features:

- `study_hours` — average hours spent studying per day
- `attendance` — attendance percentage
- `previous_score` — score in a previous examination
- `sleep_hours` — average hours of sleep per day
- `assignments_completed` — number of assignments completed

The workflow includes loading and preprocessing data with Pandas, splitting the data into training and test sets, training a Scikit-learn regression model, evaluating its performance, and visualizing the results with Matplotlib.

> **Dataset note:** The included CSV is a reproducible, synthetically generated sample dataset created for this portfolio project. It is not real student or institutional data. Results demonstrate the workflow and should not be presented as evidence of real-world predictive accuracy.

## Repository structure

```text
student-performance-prediction/
├── data/
│   └── student_performance.csv
├── models/
│   └── student_score_model.pkl
├── outputs/
│   ├── metrics.txt
│   └── figures/
│       ├── actual_vs_predicted.png
│       ├── feature_coefficients.png
│       └── residuals.png
├── src/
│   ├── train_model.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Getting started

### 1. Install Python

Use Python 3.9 or later.

### 2. Install dependencies

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train_model.py
```

This command trains the model, prints evaluation metrics, saves the trained model in `models/`, and generates charts in `outputs/figures/`.

### 4. Make a sample prediction

After training, run:

```bash
python src/predict.py
```

Edit the example student's values inside `src/predict.py` to test a different profile.

## Model evaluation

The model is evaluated on a held-out test set using:

| Metric | Meaning |
|---|---|
| MAE | Average absolute difference between actual and predicted scores |
| MSE | Average squared prediction error |
| RMSE | Square root of MSE, expressed in score units |
| R² | Proportion of target variance explained by the model |

The exact results are generated when you run the training script and saved to `outputs/metrics.txt`.

## Visualizations

The project generates:

1. **Actual vs Predicted:** compares model predictions with actual test-set scores.
2. **Residual Distribution:** shows the distribution of prediction errors.
3. **Feature Coefficients:** displays the learned Linear Regression coefficients.

## Skills demonstrated

- Python
- Pandas and NumPy
- Data cleaning and preprocessing
- Scikit-learn
- Linear Regression
- Train-test split
- Regression evaluation metrics
- Matplotlib data visualization
- Saving and loading a trained ML model

## Possible future improvements

- Use a real, properly licensed student-performance dataset.
- Compare Linear Regression with Decision Tree and Random Forest regressors.
- Add cross-validation and hyperparameter tuning.
- Build a Streamlit interface for interactive predictions.

## Disclaimer

This is an educational portfolio project. Predictions are estimates, not guarantees of academic performance. The synthetic dataset is intended only to demonstrate a machine-learning workflow.
