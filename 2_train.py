import numpy as np
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (precision_recall_curve, confusion_matrix,
                             average_precision_score, recall_score,
                             precision_score, fbeta_score)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd

OUTPUT_DIR = 'model/'
RANDOM_STATE = 42
N_FOLDS = 3

COMMON = ['F115','F321','F527','F531','F670','F1692','F2082','F2122',
          'F2582','F2678','F2737','F2956','F3043','F3836','F3887',
          'F3889','F3891','F3894']
CAT_COLS = ['F3886','F3890','F3891','F3892','F3893']
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


def preprocess(df):
    df = df.copy()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    nunique = df.nunique(dropna=False)
    df = df.drop(columns=nunique[nunique == 1].index)
    null_pct = df.isnull().mean()
    high_missing = [c for c in null_pct[null_pct > 0.70].index
                    if c not in COMMON and c != 'F3924']
    df = df.drop(columns=high_missing)
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
        d = df['F3889'].str.extract(r'^([A-Za-z]+)', expand=False)
        df['F3889_type'] = d.map(F3889_TYPE_MAP).fillna(0).astype(int)
        df['F3889_days'] = df['F3889'].str.extract(r'(\d+)', expand=False).astype(float)
        df = df.drop(columns=['F3889'])
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna('MISSING')
            uniq = df[c].nunique()
            if uniq <= 10:
                dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
            else:
                df[f'{c}_target_enc'] = df.groupby(c)['F3924'].transform('mean')
            df = df.drop(columns=[c])
    for c in df.columns:
        if c == 'F3924': continue
        if df[c].dtype in ['float64','int64','float32','int32']:
            df[c] = df[c].fillna(df[c].median())
    return df


def select_features(X, y):
    common_final = [c for c in COMMON if c != 'F3889'] + ['F3889_type','F3889_days']
    common_present = [c for c in common_final if c in X.columns]
    other = [c for c in X.columns if c not in common_present]
    X_other = X[other].select_dtypes(include=[np.number]).fillna(0)
    n_available = min(X_other.shape[1], 82)
    if n_available <= 0:
        return common_present
    selector = SelectKBest(mutual_info_classif, k=n_available)
    selector.fit(X_other, y)
    selected_mask = selector.get_support()
    selected_other = [X_other.columns[i] for i in range(len(X_other.columns)) if selected_mask[i]]
    return common_present + selected_other


def compute_scale_pos_weight(y):
    neg, pos = (y == 0).sum(), (y == 1).sum()
    return neg / pos


def train_xgboost(X, y):
    scale_pos_weight = compute_scale_pos_weight(y)
    print(f'Scale pos weight: {scale_pos_weight:.2f}')
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 0.67],
        'min_child_weight': [1, 5],
    }
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=xgb_model, param_grid=param_grid,
        scoring='average_precision', cv=cv, n_jobs=-1, verbose=1,
    )
    grid.fit(X, y)
    print(f'\nBest params: {grid.best_params_}')
    print(f'Best CV average precision: {grid.best_score_:.4f}')
    return grid.best_estimator_


def compute_shap(model, X, feature_names):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_ranking = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])
        print('\n=== SHAP Feature Importance (Top 20) ===')
        for name, val in shap_ranking[:20]:
            print(f'  {name}: {val:.6f}')
        return shap_ranking, explainer, shap_values
    except Exception as e:
        print(f'SHAP computation failed: {e}')
        return [], None, None


def evaluate_test(model, features):
    if not __import__('os').path.exists('test.csv'):
        print('\nNo test.csv found — skipping test evaluation.')
        return None
    print('\n=== Evaluating on held-out test set ===')
    test = pd.read_csv('test.csv')
    test_clean = preprocess(test)
    X_test = test_clean.drop(columns=['F3924'])[features].fillna(0).values
    y_test = test_clean['F3924'].values
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return {
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': (y_pred == y_test).mean(),
        'recall': float(recall_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'f2': float(fbeta_score(y_test, y_pred, beta=2)),
        'pr_auc': float(average_precision_score(y_test, y_prob)),
    }


def main():
    print('Loading training data...')
    train = pd.read_csv('train.csv')
    train = preprocess(train)
    y = train['F3924'].values
    X = train.drop(columns=['F3924'])
    features = select_features(X, y)
    X_arr = X[features].fillna(0).values
    print(f'X: {X_arr.shape}, y: {y.shape}')
    print(f'Target: {y.sum()} positive, {len(y) - y.sum()} negative')

    model = train_xgboost(X_arr, y)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y))
    params = model.get_params()
    params.pop('random_state', None)
    params.pop('n_jobs', None)
    params.pop('verbosity', None)
    for train_idx, val_idx in cv.split(X_arr, y):
        m = xgb.XGBClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
        m.fit(X_arr[train_idx], y[train_idx])
        oof_probs[val_idx] = m.predict_proba(X_arr[val_idx])[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y, oof_probs)
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f'\nOptimal F2 threshold (from CV): {best_threshold:.4f}')

    shap_ranking, explainer, shap_values = compute_shap(model, X_arr, features)

    test_eval = evaluate_test(model, features)

    model.save_model(f'{OUTPUT_DIR}xgb_model.json')
    joblib.dump(best_threshold, f'{OUTPUT_DIR}threshold.pkl')
    joblib.dump(features, f'{OUTPUT_DIR}selected_features.pkl')
    if explainer is not None:
        joblib.dump(explainer, f'{OUTPUT_DIR}shap_explainer.pkl')
    if test_eval is not None:
        joblib.dump(test_eval, f'{OUTPUT_DIR}test_eval.pkl')
    print(f'\nModel & artifacts saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
