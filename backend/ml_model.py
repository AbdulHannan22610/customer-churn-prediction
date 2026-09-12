"""Train and persist the customer churn Random Forest model for Railway."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "demo_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    categorical_columns = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [column for column in features.columns if column not in categorical_columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data was not found at {DATA_PATH}. Add backend/data/demo_dataset.csv first.")
    data = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError("The dataset must contain a 'churn' column.")
    target = data[TARGET_COLUMN].astype(str).str.strip().str.lower().map({"yes": "Yes", "no": "No"})
    if target.isna().any() or target.nunique() < 2:
        raise ValueError("The churn column must contain both Yes and No values.")
    feature_columns = [column for column in data.columns if column not in {TARGET_COLUMN, ID_COLUMN}]
    if not feature_columns:
        raise ValueError("No predictive feature columns were found.")
    return data[feature_columns].copy(), target


def train_model() -> None:
    features, target = load_training_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.30, random_state=42, stratify=target
    )
    pipeline = build_pipeline(features)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    print("Customer Churn Prediction - Model Evaluation")
    print("=" * 48)
    print(f"Training rows: {len(x_train)}")
    print(f"Testing rows:  {len(x_test)}")
    print(f"Accuracy:      {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision:     {precision_score(y_test, predictions, pos_label='Yes', zero_division=0):.4f}")
    print(f"Recall:        {recall_score(y_test, predictions, pos_label='Yes', zero_division=0):.4f}")
    print(f"F1 Score:      {f1_score(y_test, predictions, pos_label='Yes', zero_division=0):.4f}")
    print(classification_report(y_test, predictions, zero_division=0))
    confusion = confusion_matrix(y_test, predictions, labels=["No", "Yes"])
    print("Confusion matrix (rows = actual, columns = predicted; No, Yes):")
    print(confusion)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]
    print(pd.Series(classifier.feature_importances_, index=feature_names).nlargest(10).to_string())
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": features.columns.tolist(),
            "feature_names": feature_names.tolist(),
            "target_column": TARGET_COLUMN,
            "id_column": ID_COLUMN,
        },
        MODEL_PATH,
    )
    print(f"Saved trained model to {MODEL_PATH}")
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="YlGnBu", cbar=False, xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=axis)
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_model()
