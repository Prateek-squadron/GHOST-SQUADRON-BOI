import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_curve, average_precision_score)

st.set_page_config(page_title="Mule Account Detection", page_icon="",
                   layout="wide", initial_sidebar_state="expanded")

OUTPUT_DIR = 'model/'
CAT_COLS = ['F3886', 'F3890', 'F3891', 'F3892', 'F3893']
DATE_COL = 'F3888'
MONTH_COL = 'F2230'
TARGET = 'F3924'
MONTH_MAP = {
    'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
    'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12,
    'Jan25':1,'Feb25':2,'Mar25':3,'Apr25':4,'May25':5,
    'Jun25':6,'Jul25':7,'Aug25':8,'Sep25':9,'Oct25':10,
    'Nov25':11,'Dec25':12,
}
F3889_TYPE_MAP = {'G':0,'L':1}
sns.set_style('whitegrid')


@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(f'{OUTPUT_DIR}xgb_model.json')
    threshold = joblib.load(f'{OUTPUT_DIR}threshold.pkl')
    features = joblib.load(f'{OUTPUT_DIR}selected_features.pkl')
    explainer = joblib.load(f'{OUTPUT_DIR}shap_explainer.pkl')
    test_eval = joblib.load(f'{OUTPUT_DIR}test_eval.pkl')
    return model, threshold, features, explainer, test_eval


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
        s = df['F3889'].astype(str)
        df['F3889_type'] = s.str.extract(r'^([A-Za-z]+)', expand=False).map(F3889_TYPE_MAP).fillna(0).astype(int)
        df['F3889_days'] = pd.to_numeric(s.str.extract(r'(\d+)', expand=False), errors='coerce')
        df = df.drop(columns=['F3889'])
    for col in CAT_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Legitimate', 'Suspicious'])
    disp.plot(cmap='Blues', ax=ax, values_format='d', colorbar=False)
    ax.set_title('Confusion Matrix (Test Set)', fontsize=13)
    plt.tight_layout()
    return fig


def plot_pr_curve(y_test, y_prob):
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recalls, precisions, color='#2196F3', linewidth=2.5, label=f'PR-AUC = {ap:.3f}')
    ax.fill_between(recalls, precisions, alpha=0.15, color='#2196F3')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=13)
    ax.legend(loc='lower left')
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    return fig


def plot_shap_summary(explainer, X_sample, feature_names):
    shap_values = explainer.shap_values(X_sample)
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=15, alpha=0.7)
    plt.tight_layout()
    return fig


def plot_shap_bar(explainer, X_sample, feature_names):
    shap_values = explainer.shap_values(X_sample)
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type='bar', show=False, max_display=15)
    plt.tight_layout()
    return fig


def plot_cv_metrics():
    cv_data = pd.DataFrame({
        'Fold': ['1', '2', '3', '4', '5'],
        'Recall': [0.923, 1.000, 0.923, 1.000, 1.000],
        'Precision': [1.000, 0.929, 0.923, 1.000, 0.867],
        'F2': [0.938, 0.985, 0.923, 1.000, 0.970],
    }).melt(id_vars='Fold', var_name='Metric', value_name='Score')
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=cv_data, x='Fold', y='Score', hue='Metric',
                palette='muted', edgecolor='black', ax=ax)
    ax.set_ylim(0.8, 1.05)
    ax.set_title('5-Fold Cross-Validation Performance', fontsize=13)
    ax.legend(loc='lower right')
    plt.tight_layout()
    return fig


def plot_risk_distribution(y_test, y_prob):
    scores = (y_prob * 100).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores[y_test == 0], bins=20, alpha=0.6, label='Legitimate',
            color='#4CAF50', edgecolor='black')
    ax.hist(scores[y_test == 1], bins=20, alpha=0.85, label='Suspicious',
            color='#F44336', edgecolor='black')
    ax.set_xlabel('Risk Score (0–100)')
    ax.set_ylabel('Count')
    ax.set_title('Risk Score Distribution by Actual Class', fontsize=13)
    ax.legend()
    plt.tight_layout()
    return fig


def shap_waterfall(shap_val, instance, feature_names, expected_val):
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap.Explanation(
        values=shap_val, base_values=expected_val,
        data=instance, feature_names=feature_names
    ), max_display=10, show=False)
    plt.tight_layout()
    return fig


def main():
    st.title("Mule Account Detection System")
    st.markdown("AI/ML-powered classification of suspicious mule accounts using XGBoost")

    try:
        model, threshold, selected_features, explainer, test_eval = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Run `python3 2_train.py` first to train the model.")
        return

    tab1, tab2, tab3 = st.tabs([" Batch Predict", " Single Predict", " Model Insights"])

    with tab1:
        st.subheader("Batch Prediction")
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

                st.success(f"Processed {len(out)} accounts")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Accounts", len(out))
                c2.metric("Flagged Suspicious", int(preds.sum()),
                          delta=f"{preds.mean()*100:.1f}%")
                c3.metric("Threshold", f"{threshold:.3f}")

                st.dataframe(out[['risk_score', 'prediction', 'probability']],
                             use_container_width=True, hide_index=True)
                csv = out.to_csv(index=False)
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

                st.subheader("Risk Score Distribution")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(scores[preds == 0], bins=20, alpha=0.6,
                        label='LEGITIMATE', color='#4CAF50', edgecolor='black')
                ax.hist(scores[preds == 1], bins=20, alpha=0.85,
                        label='SUSPICIOUS', color='#F44336', edgecolor='black')
                ax.set_xlabel('Risk Score (0–100)')
                ax.set_ylabel('Count')
                ax.set_title('Predicted Risk Scores')
                ax.legend()
                st.pyplot(fig)

                if preds.sum() > 0:
                    st.subheader("SHAP Explanations for Flagged Accounts")
                    flagged_X = X[preds == 1]
                    with st.spinner("Computing SHAP..."):
                        sv = explainer.shap_values(flagged_X)
                    idx = st.selectbox("Select flagged account:", range(len(flagged_X)),
                                       format_func=lambda i: f"Account #{i+1}")
                    st.pyplot(shap_waterfall(sv[idx], flagged_X[idx],
                                              selected_features, explainer.expected_value))

    with tab2:
        st.subheader("Single Account Prediction")
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

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Score", f"{score}/100",
                      delta="HIGH" if pred else "LOW",
                      delta_color="inverse" if pred else "normal")
            c2.metric("Prediction", "SUSPICIOUS" if pred else "LEGITIMATE")
            c3.metric("Confidence", f"{proba:.2%}")

            if pred:
                sv = explainer.shap_values(X)[0]
                st.subheader("Top Contributing Features")
                contrib = sorted(zip(selected_features, sv), key=lambda x: -abs(x[1]))
                for feat, val in contrib[:5]:
                    icon = "" if val > 0 else ""
                    st.markdown(f"- **{feat}**: {val:+.4f} {icon}")

    with tab3:
        st.subheader("Model Performance")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall", f"{test_eval['recall']:.3f}")
        c2.metric("Precision", f"{test_eval['precision']:.3f}")
        c3.metric("F2 Score", f"{test_eval['f2']:.3f}")
        c4.metric("PR-AUC", f"{test_eval['pr_auc']:.3f}")

        st.markdown(f"**Threshold:** `{threshold:.4f}` | **Features:** `{len(selected_features)}`")

        g1, g2 = st.columns(2)
        with g1:
            st.pyplot(plot_confusion_matrix(test_eval['y_test'], test_eval['y_pred']))
        with g2:
            st.pyplot(plot_pr_curve(test_eval['y_test'], test_eval['y_prob']))

        st.pyplot(plot_risk_distribution(test_eval['y_test'], test_eval['y_prob']))

        st.subheader("Feature Importance")
        g3, g4 = st.columns(2)
        with g3:
            sample_n = min(500, len(test_eval['y_test']))
            X_sample = test_eval['y_prob'][:sample_n].reshape(-1, 1)
            dummy = np.random.seed(42)
            existing_data_path = f'{OUTPUT_DIR}preprocessed_data.pkl'
            preproc = joblib.load(existing_data_path)
            X_full = preproc[0]
            sample_idx = np.random.RandomState(42).choice(len(X_full),
                          size=min(200, len(X_full)), replace=False)
            X_samp = X_full[sample_idx]
            with st.spinner("Computing SHAP summary..."):
                st.pyplot(plot_shap_summary(explainer, X_samp, selected_features))
        with g4:
            with st.spinner("Computing SHAP bar chart..."):
                st.pyplot(plot_shap_bar(explainer, X_samp, selected_features))

        st.subheader("Cross-Validation")
        st.pyplot(plot_cv_metrics())
        cv_data = pd.DataFrame({
            'Fold': ['1', '2', '3', '4', '5'],
            'Recall': [0.923, 1.000, 0.923, 1.000, 1.000],
            'Precision': [1.000, 0.929, 0.923, 1.000, 0.867],
            'F2': [0.938, 0.985, 0.923, 1.000, 0.970],
            'PR-AUC': [0.944, 0.975, 0.975, 1.000, 0.982],
        }).set_index('Fold')
        st.dataframe(cv_data, use_container_width=True)


if __name__ == '__main__':
    main()
