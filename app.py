import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_curve, average_precision_score)

st.set_page_config(page_title="GHOST — Mule Account Detection", page_icon="",
                   layout="wide", initial_sidebar_state="expanded")

OUTPUT_DIR = 'model/'
CAT_COLS = ['F3886', 'F3890', 'F3891', 'F3892', 'F3893']
DATE_COL = 'F3888'
MONTH_COL = 'F2230'
MONTH_MAP = {
    'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
    'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12,
    'Jan25':1,'Feb25':2,'Mar25':3,'Apr25':4,'May25':5,
    'Jun25':6,'Jul25':7,'Aug25':8,'Sep25':9,'Oct25':10,
    'Nov25':11,'Dec25':12,
}
F3889_TYPE_MAP = {'G':0,'L':1}

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.8rem; }
[data-testid="stMetricDelta"] { font-size: 0.9rem; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; }
.stTabs [data-baseweb="tab"] { padding: 8px 24px; font-size: 0.95rem; }
div[data-testid="stExpander"] div[role="button"] p { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(f'{OUTPUT_DIR}xgb_model.json')
    threshold = joblib.load(f'{OUTPUT_DIR}threshold.pkl')
    features = joblib.load(f'{OUTPUT_DIR}selected_features.pkl')
    explainer = joblib.load(f'{OUTPUT_DIR}shap_explainer.pkl')
    test_eval = joblib.load(f'{OUTPUT_DIR}test_eval.pkl')
    shap_sample = joblib.load(f'{OUTPUT_DIR}shap_sample.pkl')
    return model, threshold, features, explainer, test_eval, shap_sample


def preprocess_batch(df):
    df = df.copy()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'F3924' in df.columns:
        df = df.drop(columns=['F3924'])
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
        s = df['F3889'].astype(str)
        df['F3889_type'] = s.str.extract(r'^([A-Za-z]+)', expand=False).map(F3889_TYPE_MAP).fillna(0).astype(int)
        df['F3889_days'] = pd.to_numeric(s.str.extract(r'(\d+)', expand=False), errors='coerce')
        df = df.drop(columns=['F3889'])
    for col in CAT_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def shap_waterfall(shap_val, instance, feature_names, expected_val):
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap.Explanation(
        values=shap_val, base_values=expected_val,
        data=instance, feature_names=feature_names
    ), max_display=10, show=False)
    plt.tight_layout()
    return fig


def plot_cv_chart():
    cv = pd.DataFrame({
        'Fold': ['1', '2', '3', '4', '5'],
        'Recall': [0.923, 1.000, 0.923, 1.000, 1.000],
        'Precision': [1.000, 0.929, 0.923, 1.000, 0.867],
        'F2': [0.938, 0.985, 0.923, 1.000, 0.970],
    }).melt(id_vars='Fold', var_name='Metric', value_name='Score')
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {'Recall': '#2196F3', 'Precision': '#4CAF50', 'F2': '#FF9800'}
    for metric in ['Recall', 'Precision', 'F2']:
        data = cv[cv['Metric'] == metric]
        ax.plot(data['Fold'], data['Score'], marker='o', label=metric,
                color=colors[metric], linewidth=2.5, markersize=8)
    ax.set_ylim(0.8, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('5-Fold Cross-Validation', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    try:
        model, threshold, selected_features, explainer, test_eval, shap_sample = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Run `python3 2_train.py` first to train the model.")
        return

    with st.sidebar:
        st.markdown("## GHOST")
        st.markdown("*Guarding Hostile Obfuscated Suspicious Transactions*")
        st.markdown("Mule Account Detection")
        st.divider()
        st.markdown("### Performance (CV Avg)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Recall", "97%")
        m2.metric("Precision", "94%")
        m3.metric("F2", "0.963")
        st.divider()
        st.markdown("### Top 5 Signals")
        tops = ['F3912', 'F2230_num', 'F3898', 'F2030', 'F2956']
        for i, t in enumerate(tops, 1):
            st.markdown(f"**{i}.** `{t}`")
        st.divider()
        st.caption("Threshold: `{:.4f}` | {} features".format(threshold, len(selected_features)))

    tab1, tab2, tab3 = st.tabs([" Batch Predict", " Single Predict", " Model Insights"])

    with tab1:
        st.subheader("Upload & Detect")
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            with st.spinner("Processing..."):
                df_raw = pd.read_csv(uploaded_file)
                df_proc = preprocess_batch(df_raw)
                for col in set(selected_features) - set(df_proc.columns):
                    df_proc[col] = 0.0
                X = df_proc[selected_features].fillna(0.0).values
                probas = model.predict_proba(X)[:, 1]
                preds = (probas >= threshold).astype(int)
                scores = np.round(probas * 100).astype(int)

                out = df_raw.copy()
                out['risk_score'] = scores
                out['prediction'] = ['SUSPICIOUS' if p else 'LEGITIMATE' for p in preds]
                out['probability'] = probas.round(4)

                total = len(out)
                flagged = int(preds.sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Accounts", f"{total:,}")
                c2.metric("Flagged Suspicious", f"{flagged:,}",
                          delta=f"{flagged/total*100:.1f}%")
                c3.metric("Clean Accounts", f"{total - flagged:,}")

                st.dataframe(out[['risk_score', 'prediction', 'probability']],
                             use_container_width=True, hide_index=True)
                st.download_button("Download Results", out.to_csv(index=False),
                                   "predictions.csv", "text/csv")

                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(scores[preds == 0], bins=20, alpha=0.6,
                        label='Legitimate', color='#4CAF50', edgecolor='black')
                ax.hist(scores[preds == 1], bins=20, alpha=0.85,
                        label='Suspicious', color='#F44336', edgecolor='black')
                ax.set_xlabel('Risk Score')
                ax.set_ylabel('Count')
                ax.legend()
                st.pyplot(fig)

                if flagged > 0:
                    st.subheader("Why Were These Flagged?")
                    flagged_X = X[preds == 1]
                    with st.spinner("Computing explanations..."):
                        sv = explainer.shap_values(flagged_X)
                    idx = st.selectbox("Select flagged account:",
                                       range(len(flagged_X)),
                                       format_func=lambda i: f"Account #{i+1}")
                    st.pyplot(shap_waterfall(sv[idx], flagged_X[idx],
                                              selected_features,
                                              explainer.expected_value))

    with tab2:
        st.subheader("Single Account Check")
        col1, col2 = st.columns(2)
        row = {}
        half = len(selected_features) // 2
        with col1:
            for feat in selected_features[:half]:
                row[feat] = st.text_input(f"{feat}", "", key=f"s{feat}")
        with col2:
            for feat in selected_features[half:]:
                row[feat] = st.text_input(f"{feat}", "", key=f"s{feat}")

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

            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Score", f"{score}/100",
                      delta="HIGH" if pred else "LOW",
                      delta_color="inverse" if pred else "normal")
            c2.metric("Decision", "SUSPICIOUS" if pred else "LEGITIMATE")
            c3.metric("Confidence", f"{proba:.2%}")

            if pred:
                sv = explainer.shap_values(X)[0]
                st.markdown("#### Top Contributing Factors")
                contrib = sorted(zip(selected_features, sv),
                                 key=lambda x: -abs(x[1]))
                for feat, val in contrib[:5]:
                    arrow = " increases" if val > 0 else " decreases"
                    st.markdown(f"- `{feat}` {val:+.4f} ({arrow.strip()} risk)")

    with tab3:
        st.subheader("Model Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall", f"{test_eval['recall']:.0%}")
        c2.metric("Precision", f"{test_eval['precision']:.0%}")
        c3.metric("F2 Score", f"{test_eval['f2']:.3f}")
        c4.metric("PR-AUC", f"{test_eval['pr_auc']:.3f}")

        st.markdown(f"**Threshold:** `{threshold:.4f}`  ·  **Features:** `{len(selected_features)}`")

        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(test_eval['y_test'], test_eval['y_pred'])
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ConfusionMatrixDisplay(cm, display_labels=['Legitimate', 'Suspicious']).plot(
            cmap='Blues', ax=ax, values_format='d', colorbar=False)
        ax.set_title('')
        st.pyplot(fig)

        st.markdown("#### Cross-Validation (5-Fold)")
        st.pyplot(plot_cv_chart())
        cv_data = pd.DataFrame({
            'Fold': ['1', '2', '3', '4', '5'],
            'Recall': [0.923, 1.000, 0.923, 1.000, 1.000],
            'Precision': [1.000, 0.929, 0.923, 1.000, 0.867],
            'F2': [0.938, 0.985, 0.923, 1.000, 0.970],
            'PR-AUC': [0.944, 0.975, 0.975, 1.000, 0.982],
        }).set_index('Fold')
        st.dataframe(cv_data, use_container_width=True)

        st.markdown("#### Feature Importance (SHAP)")
        with st.spinner("Rendering..."):
            sv = explainer.shap_values(shap_sample)
            fig, ax = plt.subplots(figsize=(9, 5))
            shap.summary_plot(sv, shap_sample, feature_names=selected_features,
                              plot_type='bar', show=False, max_display=15)
            plt.tight_layout()
            st.pyplot(fig)


if __name__ == '__main__':
    main()
