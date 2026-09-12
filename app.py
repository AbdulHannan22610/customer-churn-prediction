"""Streamlit dashboard for the pre-trained customer churn model."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
STYLES_PATH = PROJECT_ROOT / "styles.css"
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.70

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_styles() -> None:
    if STYLES_PATH.exists():
        st.markdown(
            f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


@st.cache_resource
def load_model_bundle() -> dict:
    return joblib.load(MODEL_PATH)


def risk_level(probability: float) -> str:
    if probability <= LOW_RISK_MAX:
        return "Low Risk"
    if probability <= MEDIUM_RISK_MAX:
        return "Medium Risk"
    return "High Risk"


def humanize_feature(feature_name: str) -> str:
    readable = feature_name.replace("categorical__", "").replace("numeric__", "")
    readable = readable.replace("_", " ").replace("-", " ")
    return readable.strip().title()


def contract_label(value: object) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    normalized = normalized.replace(" ", "-")
    labels = {
        "month-to-month": "Month-to-Month",
        "monthly": "Month-to-Month",
        "one-year": "One Year",
        "one-year-contract": "One Year",
        "two-year": "Two Year",
        "two-year-contract": "Two Year",
    }
    return labels.get(normalized, str(value))


def validate_upload(data: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [column for column in feature_columns if column not in data.columns]


def prediction_results(
    data: pd.DataFrame, bundle: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]
    model_features = data[feature_columns].copy()
    predictions = pipeline.predict(model_features)
    probabilities = pipeline.predict_proba(model_features)
    classes = list(pipeline.classes_)
    yes_index = classes.index("Yes")
    churn_probabilities = probabilities[:, yes_index]

    customer_ids = (
        data["customer_id"].astype(str)
        if "customer_id" in data.columns
        else pd.Series(
            [f"ROW-{number:04d}" for number in range(1, len(data) + 1)],
            index=data.index,
        )
    )
    results = pd.DataFrame(
        {
            "customer_id": customer_ids.to_numpy(),
            "churn_prediction": predictions,
            "churn_probability": churn_probabilities,
        }
    )
    results["risk_level"] = results["churn_probability"].map(risk_level)
    return results, model_features


def make_recommendations(row: pd.Series, top_factors: list[str], data: pd.DataFrame) -> list[str]:
    recommendations: list[str] = []
    normalized_contract = str(row.get("contract", "")).strip().lower()
    if "month" in normalized_contract:
        recommendations.append(
            "Offer a discounted one-year or two-year contract that gives the customer a clear reason to commit."
        )

    monthly_value = pd.to_numeric(pd.Series([row.get("monthly_charges")]), errors="coerce").iloc[0]
    monthly_values = pd.to_numeric(data.get("monthly_charges", pd.Series(dtype=float)), errors="coerce")
    if pd.notna(monthly_value) and monthly_values.notna().any() and monthly_value >= monthly_values.median():
        recommendations.append(
            "Review the current monthly price and offer a plan or discount that better matches the customer's usage."
        )

    tenure = pd.to_numeric(pd.Series([row.get("tenure")]), errors="coerce").iloc[0]
    if pd.notna(tenure) and tenure < 12:
        recommendations.append(
            "Schedule an early-tenure check-in and provide onboarding support before the first-year decision point."
        )

    service_columns = [
        column for column in row.index if any(
            keyword in column.lower() for keyword in ["support", "security", "backup", "protection"]
        )
    ]
    missing_services = [
        column for column in service_columns if str(row[column]).strip().lower() in {"no", "none", "no internet service"}
    ]
    if missing_services:
        recommendations.append(
            "Offer a relevant support or protection add-on and explain how it can reduce service friction."
        )

    payment_method = str(row.get("payment_method", "")).lower()
    if "electronic check" in payment_method:
        recommendations.append(
            "Offer a simpler automatic payment option and explain its convenience during the retention outreach."
        )

    for factor in top_factors:
        if len(recommendations) >= 3:
            break
        recommendations.append(
            f"Review the customer's {humanize_feature(factor).lower()} because it is a leading model factor for this prediction."
        )

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


def shap_factors(pipeline, model_features: pd.DataFrame, row_index: int) -> tuple[list[str], str]:
    transformed = pipeline.named_steps["preprocessor"].transform(model_features.iloc[[row_index]])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)
    yes_index = list(classifier.classes_).index("Yes")

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[yes_index])[0]
    elif np.asarray(shap_values).ndim == 3:
        values = np.asarray(shap_values)[0, :, yes_index]
    else:
        values = np.asarray(shap_values)[0]

    order = np.argsort(np.abs(values))[::-1]
    factors = [str(feature_names[index]) for index in order[:5]]
    directions = [
        f"{humanize_feature(feature_names[index])}: {'increases' if values[index] >= 0 else 'reduces'} churn likelihood"
        for index in order[:5]
    ]
    return factors, "; ".join(directions)


def portfolio_insights(data: pd.DataFrame, results: pd.DataFrame) -> list[str]:
    """Return short business observations derived from the current predictions."""
    insights: list[str] = []
    risk_counts = results["risk_level"].value_counts()
    if not risk_counts.empty:
        leading_risk = risk_counts.idxmax()
        leading_count = int(risk_counts.iloc[0])
        insights.append(f"{leading_risk} is the largest segment, with {leading_count:,} customers in the uploaded portfolio.")

    if "contract" in data.columns:
        contract_frame = pd.DataFrame(
            {
                "contract": data["contract"].map(contract_label),
                "churn": results["churn_prediction"].eq("Yes").to_numpy(),
            }
        )
        contract_rates = contract_frame.groupby("contract")["churn"].mean().sort_values(ascending=False)
        if not contract_rates.empty:
            contract_name = str(contract_rates.index[0])
            insights.append(f"{contract_name} customers have the highest predicted churn rate across contract groups.")

    if "monthly_charges" in data.columns:
        monthly_charges = pd.to_numeric(data["monthly_charges"], errors="coerce")
        if monthly_charges.notna().any():
            insights.append(f"The median monthly charge in this portfolio is ${monthly_charges.median():,.2f}.")

    return insights[:3]


def render_charts(data: pd.DataFrame, results: pd.DataFrame, bundle: dict) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.9)
    chart_one, chart_two = st.columns(2)

    with chart_one:
        with st.container(border=True):
            st.markdown('<div class="chart-kicker">Portfolio signal</div><div class="chart-description">Predicted customers likely to leave versus remain.</div>', unsafe_allow_html=True)
            figure, axis = plt.subplots(figsize=(5.5, 3.7))
            churn_counts = results["churn_prediction"].value_counts().reindex(["Yes", "No"], fill_value=0)
            axis.pie(
                churn_counts.values,
                labels=["Churn", "No Churn"],
                autopct="%1.1f%%",
                startangle=90,
                colors=["#c94c52", "#087f73"],
            )
            axis.set_title("Churn Distribution")
            st.pyplot(figure, clear_figure=True)

    with chart_two:
        with st.container(border=True):
            st.markdown('<div class="chart-kicker">Risk segmentation</div><div class="chart-description">A view of the portfolio across the three risk bands.</div>', unsafe_allow_html=True)
            figure, axis = plt.subplots(figsize=(5.5, 3.7))
            risk_counts = results["risk_level"].value_counts().reindex(
                ["Low Risk", "Medium Risk", "High Risk"], fill_value=0
            )
            sns.barplot(
                x=risk_counts.index,
                y=risk_counts.values,
                palette=["#087f73", "#c77a17", "#c94c52"],
                hue=risk_counts.index,
                legend=False,
                ax=axis,
            )
            axis.set_title("Risk Distribution")
            axis.set_xlabel("")
            axis.set_ylabel("Customers")
            axis.tick_params(axis="x", rotation=15)
            st.pyplot(figure, clear_figure=True)

    chart_three, chart_four = st.columns(2)
    with chart_three:
        with st.container(border=True):
            st.markdown('<div class="chart-kicker">Contract signal</div><div class="chart-description">Compare predicted churn rates across contract groups.</div>', unsafe_allow_html=True)
            figure, axis = plt.subplots(figsize=(5.5, 3.7))
            contract_column = "contract"
            if contract_column in data.columns:
                contract_frame = pd.DataFrame(
                    {
                        "contract": data[contract_column].map(contract_label).to_numpy(),
                        "churn": results["churn_prediction"].eq("Yes").to_numpy(),
                    }
                )
                contract_rates = contract_frame.groupby("contract")["churn"].mean().sort_values(ascending=False) * 100
                sns.barplot(
                    x=contract_rates.index,
                    y=contract_rates.values,
                    color="#087f73",
                    ax=axis,
                )
                axis.set_ylabel("Churn rate (%)")
                axis.set_xlabel("")
                axis.tick_params(axis="x", rotation=15)
            else:
                axis.text(0.5, 0.5, "No contract column in uploaded data", ha="center", va="center")
                axis.set_axis_off()
            axis.set_title("Churn by Contract")
            st.pyplot(figure, clear_figure=True)

    with chart_four:
        with st.container(border=True):
            st.markdown('<div class="chart-kicker">Model signal</div><div class="chart-description">The strongest encoded features used by the Random Forest.</div>', unsafe_allow_html=True)
            figure, axis = plt.subplots(figsize=(5.5, 4.6))
            classifier = bundle["pipeline"].named_steps["classifier"]
            feature_names = bundle["pipeline"].named_steps["preprocessor"].get_feature_names_out()
            importance = pd.Series(classifier.feature_importances_, index=feature_names).nlargest(12).sort_values()
            axis.barh([humanize_feature(name) for name in importance.index], importance.values, color="#c77a17")
            axis.set_title("Feature Importance")
            axis.set_xlabel("Importance")
            figure.tight_layout()
            st.pyplot(figure, clear_figure=True)


def main() -> None:
    load_styles()
    with st.sidebar:
        st.markdown('<div class="sidebar-kicker">Retention intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Customer Churn Prediction</div>', unsafe_allow_html=True)
        st.caption("Pre-trained Random Forest analytics")
        st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="upload-panel"><strong>Upload Customer Data</strong>'
            '<p>Upload a clean CSV to generate churn predictions and risk analysis.</p></div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("Upload Customer Data", type=["csv"])
        st.divider()
        st.markdown(
            '<div class="small-note">Risk thresholds: Low <= 30%, Medium <= 70%, High > 70%.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("How to read risk", expanded=False):
            st.markdown(
                "**Low Risk** customers are at or below 30% predicted churn probability. "
                "**Medium Risk** falls between 31% and 70%. **High Risk** is above 70%."
            )

    st.markdown(
        '<div class="hero"><div class="eyebrow">Retention intelligence</div>'
        '<h1>Predict churn.<br>Protect relationships.</h1>'
        '<p>Predict customer churn, identify high-risk customers, and discover the factors driving retention risk.</p>'
        '<div class="hero-meta"><span>Pre-trained Random Forest</span><span>Explainable predictions</span><span>Action-ready insights</span></div></div>',
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.info("Train the model first with `python ml_model.py`, then return here to upload customer data.")
        return
    if uploaded_file is None:
        st.markdown('<div class="section-label">Next step</div>', unsafe_allow_html=True)
        st.write("Upload a CSV in the sidebar to generate predictions, risk levels, and retention insights.")
        return

    try:
        bundle = load_model_bundle()
        uploaded_data = pd.read_csv(uploaded_file)
        missing_columns = validate_upload(uploaded_data, bundle["feature_columns"])
        if missing_columns:
            st.error("Missing required model features: " + ", ".join(missing_columns))
            return
        if uploaded_data.empty:
            st.error("The uploaded CSV does not contain any customer rows.")
            return

        results, model_features = prediction_results(uploaded_data, bundle)
    except Exception as error:
        st.error(f"The uploaded CSV could not be processed: {error}")
        return

    st.markdown(
        '<div class="processed-banner"><span class="status-dot"></span>'
        '<div><strong>Dataset processed successfully</strong>'
        '<p>Predictions and retention signals are ready to explore.</p></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
    total_customers = len(results)
    predicted_churn = int(results["churn_prediction"].eq("Yes").sum())
    high_risk = int(results["risk_level"].eq("High Risk").sum())
    average_probability = results["churn_probability"].mean()
    metrics = st.columns(5)
    metrics[0].metric("Total Customers", f"{total_customers:,}")
    metrics[1].metric("Predicted Churn", f"{predicted_churn:,}")
    metrics[2].metric("Predicted Churn Rate", f"{predicted_churn / total_customers:.1%}")
    metrics[3].metric("High-Risk Customers", f"{high_risk:,}")
    metrics[4].metric("Average Churn Probability", f"{average_probability:.1%}")

    st.markdown('<div class="section-label">Risk analysis</div>', unsafe_allow_html=True)
    insight_items = portfolio_insights(uploaded_data, results)
    if insight_items:
        st.markdown(
            '<div class="insight-grid">'
            + "".join(f'<div class="business-insight"><span>↗</span><p>{item}</p></div>' for item in insight_items)
            + '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">Churn insights</div>', unsafe_allow_html=True)
    render_charts(uploaded_data, results, bundle)

    st.markdown('<div class="section-label">Customer predictions</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-intro">Scan the portfolio, then open one customer for a deeper retention brief.</p>',
        unsafe_allow_html=True,
    )
    risk_options = ["Low Risk", "Medium Risk", "High Risk"]
    selected_risks = st.multiselect(
        "Filter the prediction table by risk",
        options=risk_options,
        default=risk_options,
        format_func=lambda risk: risk.replace(" Risk", ""),
    )
    filtered_results = results[results["risk_level"].isin(selected_risks)]
    st.markdown(
        f'<div class="table-status"><span>{len(filtered_results):,} of {len(results):,} customers shown</span>'
        '<span>Use the selector below for the full customer brief.</span></div>',
        unsafe_allow_html=True,
    )
    display_results = filtered_results.copy()
    display_results["churn_probability"] = display_results["churn_probability"].map(lambda value: f"{value:.1%}")
    st.dataframe(display_results, use_container_width=True, hide_index=True)
    st.markdown('<div class="section-label">Export results</div>', unsafe_allow_html=True)
    export_columns = st.columns([3, 1])
    with export_columns[0]:
        st.markdown('<p class="section-intro">Take the scored customer list into your retention workflow.</p>', unsafe_allow_html=True)
    with export_columns[1]:
        st.download_button(
            "Download CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

    st.markdown('<div class="section-label">Customer explanation</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-intro">Select a customer to understand what is driving their predicted risk.</p>', unsafe_allow_html=True)
    selected_position = st.selectbox(
        "Select a customer",
        options=range(len(results)),
        format_func=lambda position: str(results.iloc[position]["customer_id"]),
        help="Changing this selection refreshes the customer summary, SHAP explanation, and recommendations below.",
    )
    selected_result = results.iloc[selected_position]
    selected_row = uploaded_data.iloc[selected_position]
    selected_risk_class = selected_result["risk_level"].lower().replace(" ", "-")
    st.markdown(
        f'<div class="customer-summary {selected_risk_class}">'
        f'<div><span class="summary-kicker">Selected customer</span><strong>{selected_result["customer_id"]}</strong></div>'
        f'<div class="summary-risk"><span class="risk-pill">{selected_result["risk_level"]}</span>'
        f'<span class="summary-probability">{selected_result["churn_probability"]:.1%} churn probability</span></div></div>',
        unsafe_allow_html=True,
    )
    selected_columns = st.columns(4)
    selected_columns[0].metric("Customer ID", str(selected_result["customer_id"]))
    selected_columns[1].metric("Predicted Churn", selected_result["churn_prediction"])
    selected_columns[2].metric("Churn Probability", f"{selected_result['churn_probability']:.1%}")
    selected_columns[3].metric("Risk Level", selected_result["risk_level"])

    try:
        factors, explanation = shap_factors(bundle["pipeline"], model_features, selected_position)
        st.markdown('<div class="subsection-title">Why is this customer at risk?</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box"><strong>Model explanation</strong><p>{explanation}</p></div>', unsafe_allow_html=True)
    except Exception as error:
        factors = []
        st.warning(f"SHAP explanation was unavailable for this row: {error}")

    recommendations = make_recommendations(selected_row, factors, uploaded_data)
    st.markdown('<div class="section-label">Recommended actions</div>', unsafe_allow_html=True)
    recommendation_columns = st.columns(3)
    for index, recommendation in enumerate(recommendations):
        recommendation_columns[index].markdown(
            f'<div class="recommendation-card"><strong>Recommendation {index + 1}</strong>'
            f'<p>{recommendation}</p></div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
