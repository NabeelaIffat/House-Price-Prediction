from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "house_price_model.joblib"
METRICS_PATH = ROOT / "metrics.json"

FEATURES = [
    "GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF",
    "FullBath", "YearBuilt", "BedroomAbvGr"
]
TARGET = "SalePrice"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Download train.csv from Kaggle and place it in the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in train.csv: {missing}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, FEATURES)
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=12, random_state=42, n_jobs=-1
        )
    }

    results = {}
    best_name, best_rmse, best_pipeline = None, float("inf"), None

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        results[name] = {
            "MAE": round(float(mean_absolute_error(y_test, predictions)), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2_score(y_test, predictions)), 4)
        }
        if rmse < best_rmse:
            best_name, best_rmse, best_pipeline = name, rmse, pipeline

    joblib.dump(best_pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=4)

    print("\nModel comparison (lower MAE/RMSE is better; higher R2 is better):")
    print(pd.DataFrame(results).T)
    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
