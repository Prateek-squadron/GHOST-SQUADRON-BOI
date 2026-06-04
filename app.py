import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tempfile
import os

st.set_page_config(
    page_title="Mule Account Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUT_DIR = 'model/'
CAT_COLS = ['F3886', 'F3890', 'F3891', 'F3892', 'F3893']
DATE_COL = 'F3888'
MONTH_COL = 'F2230'
TARGET = 'F3924'

MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
    'Jan25': 1, 'Feb25': 2, 'Mar25': 3, 'Apr25': 4, 'May25': 5,
    'Jun25': 6, 'Jul25': 7, 'Aug25': 8, 'Sep25': 9, 'Oct25': 10,
    'Nov25': 11, 'Dec25': 12,
}
F3889_TYPE_MAP = {'G': 0, 'L': 1}


@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(f'{OUTPUT_DIR}xgb_model.json')
    threshold = joblib.load(f'{OUTPUT_DIR}threshold.pkl')
    features = joblib.load(f'{OUTPUT_DIR}selected_features.pkl')
    explainer = joblib.load(f'{OUTPUT_DIR}shap_explainer.pkl')
    return model, threshold, features, explainer


def preprocess_batch(df):
    df = df.copy()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    if DATE_COL in df.columns:
        dates = pd.to_datetime(df[DATE_COL], dayfirst=True, errors='coerce')
        df[f'{DATE_COL}_year'] = dates.dt.year
        df[f'{DATE_COL}_month'] = dates.dt.month
        df[f'{DATE_COL}_day'] = dates.dt.day
        df = df.drop(columns=[DATE_COL])

    if MONTH_COL in df.columns:
        df[f'{MONTH_COL}_num'] = df[MONTH_COL].map(MONTH_MAP).fillna(0).astype(int)
        df = df.drop(columns=[MONTH_COL])

    if 'F3889' in df.columns:
        f3889_str = df['F3889'].astype(str)
        f3889_type = f3889_str.str.extract(r'^([A-Za-z]+)', expand=False)
        df['F3889_type'] = f3889_type.map(F3889_TYPE_MAP).fillna(0).astype(int)
        f3889_days = f3889_str.str.extract(r'(\d+)', expand=False)
        df['F3889_days'] = pd.to_numeric(f3889_days, errors='coerce')
        df = df.drop(columns=['F3889'])

    for col in CAT_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def main():
    st.title("Mule Account Detection System")
    st.markdown("AI/ML-powered classification of suspicious mule accounts using XGBoost")

    try:
        model, threshold, selected_features, explainer = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Run `python3 2_train.py` first to train the model.")
        return

    tab1, tab2, tab3 = st.tabs([" Batch Predict", " Single Predict", " Model Insights"])

    with tab1:
        st.subheader("Batch Prediction")
        st.markdown("Upload a CSV file in the same format as the training data.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            with st.spinner("Processing..."):
                df_raw = pd.read_csv(uploaded_file)
                df_proc = preprocess_batch(df_raw)
                missing_cols = set(selected_features) - set(df_proc.columns)
                for col in missing_cols:
                    df_proc[col] = 0.0
                X = df_proc[selected_features].fillna(0.0).values
                probas = model.predict_proba(X)[:, 1]
                preds = (probas >= threshold).astype(int)
                scores = np.round(probas * 100).astype(int)

                out = df_raw.copy()
                out['risk_score'] = scores
                out['prediction'] = ['SUSPICIOUS' if p else 'LEGITIMATE' for p in preds]
                out['probability'] = probas.round(4)

                st.success(f"Processed {len(out)} accounts")

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Accounts", len(out))
                col2.metric("Flagged Suspicious", int(preds.sum()),
                            delta=f"{preds.mean()*100:.1f}%")
                col3.metric("Threshold", f"{threshold:.3f}")

                st.dataframe(
                    out[['risk_score', 'prediction', 'probability'] + 
                        [c for c in out.columns if c not in ['risk_score', 'prediction', 'probability', TARGET]][:5]],
                    use_container_width=True,
                    hide_index=True,
                )

                csv = out.to_csv(index=False)
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

                if preds.sum() > 0:
                    st.subheader("SHAP Explanations for Flagged Accounts")
                    flagged_X = X[preds == 1]
                    with st.spinner("Computing SHAP values..."):
                        shap_values = explainer.shap_values(flagged_X)
                    idx = st.selectbox("Select flagged account:", range(len(flagged_X)),
                                       format_func=lambda i: f"Account #{i+1}")
                    st.pyplot(shap_waterfall(shap_values[idx], flagged_X[idx],
                                              selected_features, explainer.expected_value))

    with tab2:
        st.subheader("Single Account Prediction")
        st.markdown("Enter feature values manually.")

        col1, col2 = st.columns(2)
        row = {}
        half = len(selected_features) // 2
        with col1:
            for feat in selected_features[:half]:
                row[feat] = st.text_input(f"{feat}", "", key=f"in_{feat}")
        with col2:
            for feat in selected_features[half:]:
                row[feat] = st.text_input(f"{feat}", "", key=f"in_{feat}")

        if st.button("Predict", type="primary"):
            clean = {}
            for k, v in row.items():
                if v == "" or v is None:
                    clean[k] = 0.0
                else:
                    try:
                        clean[k] = float(v)
                    except ValueError:
                        clean[k] = 0.0
            X = np.array([clean[f] for f in selected_features]).reshape(1, -1)
            proba = model.predict_proba(X)[0, 1]
            pred = int(proba >= threshold)
            score = int(round(proba * 100))

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Score", f"{score}/100",
                      delta="HIGH" if pred else "LOW",
                      delta_color="inverse" if pred else "normal")
            c2.metric("Prediction", "SUSPICIOUS" if pred else "LEGITIMATE")
            c3.metric("Confidence", f"{proba:.2%}")

            if pred:
                shap_val = explainer.shap_values(X)
                st.subheader("Top Contributing Features")
                contrib = list(zip(selected_features, shap_val[0]))
                contrib.sort(key=lambda x: -abs(x[1]))
                for feat, val in contrib[:5]:
                    direction = " increases" if val > 0 else " decreases"
                    st.markdown(f"- **{feat}**: {val:+.4f} ({direction.strip()} risk)")

    with tab3:
        st.subheader("Model Insights")

        st.markdown(f"**Optimal F2 threshold:** `{threshold:.4f}`")
        st.markdown(f"**Number of features:** `{len(selected_features)}`")

        st.markdown("#### 5-Fold Cross-Validation Performance")
        cv_data = pd.DataFrame({
            'Fold': ['1', '2', '3', '4', '5'],
            'Recall': [0.923, 1.000, 0.923, 1.000, 1.000],
            'Precision': [1.000, 0.929, 0.923, 1.000, 0.867],
            'F2': [0.938, 0.985, 0.923, 1.000, 0.970],
            'PR-AUC': [0.944, 0.975, 0.975, 1.000, 0.982],
        }).set_index('Fold')

        st.dataframe(cv_data, use_container_width=True)
        st.markdown(f"**Average:** Recall `0.969` | Precision `0.944` | F2 `0.963` | PR-AUC `0.975`")

        st.markdown("#### Held-Out Test Performance (16 suspicious / 1,817 total)")
        st.markdown("Recall: **1.000** | Precision: **1.000** | F2: **1.000** | PR-AUC: **1.000**")


def shap_waterfall(shap_val, instance, feature_names, expected_val):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap.Explanation(
        values=shap_val,
        base_values=expected_val,
        data=instance,
        feature_names=feature_names
    ), max_display=10, show=False)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    main()
