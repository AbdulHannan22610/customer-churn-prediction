"""FastAPI service for customer churn prediction."""

import io
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.70


def configured_origins() -> list[str]:
    value = os.getenv("FRONTEND_URL", "http://localhost:5500")
    return [origin.strip() for origin in value.split(",") if origin.strip()]


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


_model_bundle: dict[str, Any] | None = None


def get_model_bundle() -> dict[str, Any]:
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="The trained model is not available. Run python ml_model.py first.",
            )
        try:
            _model_bundle = joblib.load(MODEL_PATH)
        except Exception as error:
            raise HTTPException(status_code=503, detail="The trained model could not be loaded.") from error
    return _model_bundle


def risk_level(probability: float) -> str:
    if probability <= LOW_RISK_MAX:
        return "Low Risk"
    if probability <= MEDIUM_RISK_MAX:
        return "Medium Risk"
    return "High Risk"


def humanize_feature(feature_name: str) -> str:
    readable = feature_name.replace("categorical__", "").replace("numeric__", "")
    return readable.replace("_", " ").replace("-", " ").strip().title()


def contract_label(value: object) -> str:
    normalized = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    labels = {
        "month-to-month": "Month-to-Month",
        "monthly": "Month-to-Month",
        "one-year": "One Year",
        "one-year-contract": "One Year",
        "two-year": "Two Year",
        "two-year-contract": "Two Year",
    }
    return labels.get(normalized, str(value))


def make_recommendations(row: pd.Series, top_factors: list[str], data: pd.DataFrame) -> list[str]:
    recommendations: list[str] = []
    normalized_contract = str(row.get("contract", "")).strip().lower()
    if "month" in normalized_contract:
        recommendations.append("Offer a discounted one-year or two-year contract that gives the customer a clear reason to commit.")

    monthly_value = pd.to_numeric(pd.Series([row.get("monthly_charges")]), errors="coerce").iloc[0]
    monthly_values = pd.to_numeric(data.get("monthly_charges", pd.Series(dtype=float)), errors="coerce")
    if pd.notna(monthly_value) and monthly_values.notna().any() and monthly_value >= monthly_values.median():
        recommendations.append("Review the current monthly price and offer a plan or discount that better matches the customer's usage.")

    tenure = pd.to_numeric(pd.Series([row.get("tenure")]), errors="coerce").iloc[0]
    if pd.notna(tenure) and tenure < 12:
        recommendations.append("Schedule an early-tenure check-in and provide onboarding support before the first-year decision point.")

    service_columns = [column for column in row.index if any(keyword in column.lower() for keyword in ["support", "security", "backup", "protection"])]
    if any(str(row[column]).strip().lower() in {"no", "none", "no internet service"} for column in service_columns):
        recommendations.append("Offer a relevant support or protection add-on and explain how it can reduce service friction.")

    if "electronic check" in str(row.get("payment_method", "")).lower():
        recommendations.append("Offer a simpler automatic payment option and explain its convenience during the retention outreach.")

    for factor in top_factors:
        if len(recommendations) >= 3:
            break
        recommendations.append(f"Review the customer's {humanize_feature(factor).lower()} because it is a leading model factor for this prediction.")

    fallback_messages = [
        "Use the customer's current account profile to tailor the retention conversation.",
        "Confirm the customer's service needs and resolve any account friction during outreach.",
        "Set a follow-up date to measure whether the selected retention action improved engagement.",
    ]
    for message in fallback_messages:
        if len(recommendations) >= 3:
            break
        recommendations.append(message)
    return recommendations[:3]


def shap_explanations(pipeline: Any, model_features: pd.DataFrame) -> list[tuple[list[str], str]]:
    transformed = pipeline.named_steps["preprocessor"].transform(model_features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)
    yes_index = list(classifier.classes_).index("Yes")

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[yes_index])
    elif np.asarray(shap_values).ndim == 3:
        values = np.asarray(shap_values)[:, :, yes_index]
    else:
        values = np.asarray(shap_values)

    explanations: list[tuple[list[str], str]] = []
    for row_values in values:
        order = np.argsort(np.abs(row_values))[::-1][:5]
        factors = [str(feature_names[index]) for index in order]
        directions = [
            f"{humanize_feature(feature_names[index])}: {'increases' if row_values[index] >= 0 else 'reduces'} churn likelihood"
            for index in order
        ]
        explanations.append((factors, "; ".join(directions)))
    return explanations


def validate_data(data: pd.DataFrame, bundle: dict[str, Any]) -> None:
    if data.empty:
        raise ValueError("The uploaded CSV does not contain any customer rows.")
    missing = [column for column in bundle["feature_columns"] if column not in data.columns]
    if missing:
        raise ValueError("Missing required model features: " + ", ".join(missing))


def build_response(data: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]
    model_features = data[feature_columns].copy()
    predictions = pipeline.predict(model_features)
    probabilities = pipeline.predict_proba(model_features)
    yes_index = list(pipeline.classes_).index("Yes")
    churn_probabilities = probabilities[:, yes_index]
    explanations = shap_explanations(pipeline, model_features)

    customer_ids = data["customer_id"].astype(str).tolist() if "customer_id" in data.columns else [f"ROW-{index:04d}" for index in range(1, len(data) + 1)]
    result_rows: list[dict[str, Any]] = []
    for index, customer_id in enumerate(customer_ids):
        factors, explanation = explanations[index]
        result_rows.append(
            {
                "index": index,
                "customer_id": customer_id,
                "churn_prediction": str(predictions[index]),
                "churn_probability": float(churn_probabilities[index]),
                "risk_level": risk_level(float(churn_probabilities[index])),
                "shap_explanation": explanation,
                "recommendations": make_recommendations(data.iloc[index], factors, data),
            }
        )

    results = pd.DataFrame(result_rows)
    risk_order = ["Low Risk", "Medium Risk", "High Risk"]
    risk_counts = results["risk_level"].value_counts().reindex(risk_order, fill_value=0)
    churn_counts = results["churn_prediction"].value_counts().reindex(["Yes", "No"], fill_value=0)

    contract_chart = {"labels": [], "values": []}
    if "contract" in data.columns:
        contract_frame = pd.DataFrame({"contract": data["contract"].map(contract_label), "churn": results["churn_prediction"].eq("Yes")})
        contract_rates = (contract_frame.groupby("contract")["churn"].mean() * 100).sort_values(ascending=False)
        contract_chart = {"labels": contract_rates.index.tolist(), "values": [float(value) for value in contract_rates.values]}

    classifier = pipeline.named_steps["classifier"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.Series(classifier.feature_importances_, index=feature_names).nlargest(12).sort_values()

    return {
        "summary": {
            "total_customers": len(results),
            "predicted_churn": int(results["churn_prediction"].eq("Yes").sum()),
            "predicted_churn_rate": float(results["churn_prediction"].eq("Yes").mean()),
            "high_risk_customers": int(results["risk_level"].eq("High Risk").sum()),
            "average_churn_probability": float(results["churn_probability"].mean()),
        },
        "results": result_rows,
        "charts": {
            "churn_distribution": {"labels": ["Churn", "No Churn"], "values": [int(churn_counts["Yes"]), int(churn_counts["No"])]},
            "risk_distribution": {"labels": risk_order, "values": [int(value) for value in risk_counts.values]},
            "churn_by_contract": contract_chart,
            "feature_importance": {"labels": [humanize_feature(name) for name in importance.index], "values": [float(value) for value in importance.values]},
        },
        "feature_importance": {humanize_feature(name): float(value) for name, value in importance.items()},
    }


async def read_upload(upload: UploadFile) -> pd.DataFrame:
    if not upload.filename or not upload.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    try:
        content = await upload.read()
        if not content:
            raise ValueError("The uploaded CSV is empty.")
        return pd.read_csv(io.BytesIO(content))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid CSV.") from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_available": str(MODEL_PATH.exists()).lower()}


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)) -> JSONResponse:
    bundle = get_model_bundle()
    data = await read_upload(file)
    try:
        validate_data(data, bundle)
        return JSONResponse(build_response(data, bundle))
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"The CSV could not be scored: {error}") from error


@app.post("/predict")
async def predict_json(payload: dict[str, Any]) -> dict[str, Any]:
    bundle = get_model_bundle()
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="Send a non-empty 'rows' array.")
    try:
        data = pd.DataFrame(rows)
        validate_data(data, bundle)
        return build_response(data, bundle)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"The rows could not be scored: {error}") from error


@app.get("/feature-importance")
def feature_importance() -> dict[str, Any]:
    bundle = get_model_bundle()
    classifier = bundle["pipeline"].named_steps["classifier"]
    feature_names = bundle["pipeline"].named_steps["preprocessor"].get_feature_names_out()
    importance = pd.Series(classifier.feature_importances_, index=feature_names).sort_values(ascending=False)
    return {"feature_importance": {humanize_feature(name): float(value) for name, value in importance.items()}}


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "customer-churn-prediction", "docs": "/docs"}
