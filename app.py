import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Mule Account Detection", page_icon="",
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


def main():
    try:
        model, threshold, selected_features, explainer, test_eval = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Run `python3 2_train.py` first to train the model.")
        return

    with st.sidebar:
        st.markdown("## GHOST SQUADRON BOI")
        st.markdown("Mule Account Detection System")
        st.divider()
        st.markdown("### Performance")
        st.markdown(f"**Recall:** {test_eval['recall']:.0%}")
        st.markdown(f"**Precision:** {test_eval['precision']:.0%}")
        st.markdown(f"**F2 Score:** {test_eval['f2']:.3f}")
        st.markdown(f"**Threshold:** {threshold:.3f}")
        st.divider()
        st.markdown("### Top Signals")
        st.markdown("- F3912")
        st.markdown("- F2230_num (month)")
        st.markdown("- F3898")
        st.markdown("- F2030")
        st.markdown("- F2956")

    tab1, tab2 = st.tabs([" Batch Predict", " Single Predict"])

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

                st.success(f"Processed {len(out)} accounts | "
                           f"{int(preds.sum())} flagged suspicious")

                st.dataframe(out[['risk_score', 'prediction', 'probability']],
                             use_container_width=True, hide_index=True)

                csv = out.to_csv(index=False)
                st.download_button("Download Results", csv, "predictions.csv", "text/csv")

                st.subheader("Risk Score Distribution")
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(scores[preds == 0], bins=20, alpha=0.6,
                        label='Legitimate', color='#4CAF50', edgecolor='black')
                ax.hist(scores[preds == 1], bins=20, alpha=0.85,
                        label='Suspicious', color='#F44336', edgecolor='black')
                ax.set_xlabel('Risk Score')
                ax.set_ylabel('Count')
                ax.legend()
                st.pyplot(fig)

                if preds.sum() > 0:
                    st.subheader("Why These Accounts Were Flagged")
                    flagged_X = X[preds == 1]
                    with st.spinner("Computing explanations..."):
                        sv = explainer.shap_values(flagged_X)
                    idx = st.selectbox("Select a flagged account:",
                                       range(len(flagged_X)),
                                       format_func=lambda i: f"Account #{i+1}")
                    st.pyplot(shap_waterfall(sv[idx], flagged_X[idx],
                                              selected_features,
                                              explainer.expected_value))

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
                clean[k] = 0.0 if v == "" or v is None else float(v) if v.replace('.','',1).replace('-','',1).isdigit() else 0.0
            X = np.array([clean[f] for f in selected_features]).reshape(1, -1)
            proba = model.predict_proba(X)[0, 1]
            pred = int(proba >= threshold)
            score = int(round(proba * 100))

            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Score", f"{score}/100",
                      delta="HIGH" if pred else "LOW",
                      delta_color="inverse" if pred else "normal")
            c2.metric("Prediction", "SUSPICIOUS" if pred else "LEGITIMATE")
            c3.metric("Confidence", f"{proba:.2%}")

            if pred:
                sv = explainer.shap_values(X)[0]
                st.subheader("Top Contributing Factors")
                contrib = sorted(zip(selected_features, sv),
                                 key=lambda x: -abs(x[1]))
                for feat, val in contrib[:5]:
                    arrow = " ↑" if val > 0 else " ↓"
                    st.markdown(f"- **{feat}**: {val:+.4f}{arrow}")


if __name__ == '__main__':
    main()
