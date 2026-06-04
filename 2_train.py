import numpy as np
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_recall_curve

OUTPUT_DIR = 'model/'
RANDOM_STATE = 42
N_FOLDS = 3

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
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='average_precision',
        cv=cv,
        n_jobs=-1,
        verbose=1,
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
        shap_ranking = sorted(
            zip(feature_names, mean_abs_shap),
            key=lambda x: -x[1]
        )
        print('\n=== SHAP Feature Importance (Top 20) ===')
        for name, val in shap_ranking[:20]:
            print(f'  {name}: {val:.6f}')
        return shap_ranking, explainer
    except Exception as e:
        print(f'SHAP computation failed: {e}')
        return [], None

def main():
    print('Loading preprocessed data...')
    X, y, selected_features, encoders = joblib.load(
        f'{OUTPUT_DIR}preprocessed_data.pkl'
    )
    print(f'X: {X.shape}, y: {y.shape}')
    print(f'Target: {y.sum()} positive, {len(y) - y.sum()} negative')

    model = train_xgboost(X, y)

    # Find optimal F2 threshold from CV out-of-fold predictions
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y))
    params = model.get_params()
    params.pop('random_state', None)
    params.pop('n_jobs', None)
    params.pop('verbosity', None)
    for train_idx, val_idx in cv.split(X, y):
        m = xgb.XGBClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
        m.fit(X[train_idx], y[train_idx])
        oof_probs[val_idx] = m.predict_proba(X[val_idx])[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y, oof_probs)
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f'\nOptimal F2 threshold (from CV): {best_threshold:.4f}')

    shap_ranking, explainer = compute_shap(model, X, selected_features)

    model.save_model(f'{OUTPUT_DIR}xgb_model.json')
    joblib.dump(best_threshold, f'{OUTPUT_DIR}threshold.pkl')
    joblib.dump(selected_features, f'{OUTPUT_DIR}selected_features.pkl')
    if explainer is not None:
        joblib.dump(explainer, f'{OUTPUT_DIR}shap_explainer.pkl')
    print(f'\nModel & artifacts saved to {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
