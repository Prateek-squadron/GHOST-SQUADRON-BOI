import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

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


def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model(f'{OUTPUT_DIR}xgb_model.json')
    threshold = joblib.load(f'{OUTPUT_DIR}threshold.pkl')
    selected_features = joblib.load(f'{OUTPUT_DIR}selected_features.pkl')
    explainer = joblib.load(f'{OUTPUT_DIR}shap_explainer.pkl')
    return model, threshold, selected_features, explainer


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


def risk_score(probability):
    return int(round(probability * 100))


def main():
    import sys
    model, threshold, selected_features, explainer = load_artifacts()
    print(f'Loaded model | threshold={threshold:.4f} | features={len(selected_features)}')

    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        csv_path = sys.argv[2] if len(sys.argv) > 2 else input('CSV path: ')
        df = pd.read_csv(csv_path)
        df_proc = preprocess_batch(df)

        missing_cols = set(selected_features) - set(df_proc.columns)
        for col in missing_cols:
            df_proc[col] = 0.0

        X = df_proc[selected_features].fillna(0.0).values

        probas = model.predict_proba(X)[:, 1]
        preds = (probas >= threshold).astype(int)
        scores = np.round(probas * 100).astype(int)

        out = pd.DataFrame({
            'prediction': preds,
            'probability': probas.round(4),
            'risk_score': scores,
            'threshold_used': round(threshold, 4),
        })
        out['threshold_used'] = out['threshold_used'].astype(float)

        flagged_indices = np.where(preds == 1)[0]
        if len(flagged_indices) > 0:
            print(f'Computing SHAP explanations for {len(flagged_indices)} flagged accounts...')
            flagged_X = X[flagged_indices]
            shap_values = explainer.shap_values(flagged_X)
            out['top_contributors'] = ''
            for idx_in_group, orig_idx in enumerate(flagged_indices):
                contrib = list(zip(selected_features, shap_values[idx_in_group]))
                contrib.sort(key=lambda x: -abs(x[1]))
                top5 = [{'feature': f, 'shap_value': round(float(v), 4)} for f, v in contrib[:5]]
                out.at[orig_idx, 'top_contributors'] = str(top5)

        out.to_csv('predictions_output.csv', index=False)
        print(f'Batch predictions saved to predictions_output.csv')
        suspicious = int(preds.sum())
        print(f'Suspicious accounts flagged: {suspicious}/{len(out)}')

    else:
        print('\nSingle prediction mode. Enter feature values:')
        row = {}
        for col in selected_features:
            val = input(f'  {col}: ').strip()
            if val == '':
                row[col] = np.nan
            else:
                try:
                    row[col] = float(val)
                except ValueError:
                    row[col] = val if val else np.nan
        row_s = pd.Series(row).fillna(0.0)
        X = row_s[selected_features].values.reshape(1, -1)
        proba = model.predict_proba(X)[0, 1]
        pred = int(proba >= threshold)
        print(f'\nPrediction: {"SUSPICIOUS" if pred else "LEGITIMATE"}')
        print(f'Risk Score: {risk_score(proba)}/100')
        print(f'Probability: {proba:.4f}')


if __name__ == '__main__':
    main()
