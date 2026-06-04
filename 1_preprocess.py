import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'train.csv'
OUTPUT_DIR = 'model/'
RANDOM_STATE = 42

COMMON_FEATURES = [
    'F115', 'F321', 'F527', 'F531', 'F670', 'F1692', 'F2082', 'F2122',
    'F2582', 'F2678', 'F2737', 'F2956', 'F3043', 'F3836', 'F3887',
    'F3889_type', 'F3889_days', 'F3891', 'F3894'
]

CAT_COLS = ['F3886', 'F3890', 'F3891', 'F3892', 'F3893']
DATE_COL = 'F3888'
MONTH_COL = 'F2230'

TARGET = 'F3924'
MAX_FEATURES = 100

def load_data():
    df = pd.read_csv(DATA_PATH)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    print(f'Loaded: {df.shape[0]} rows, {df.shape[1]} cols')
    return df

def drop_constant_cols(df):
    cols_before = df.shape[1]
    nunique = df.nunique(dropna=False)
    constant_cols = nunique[nunique == 1].index.tolist()
    df = df.drop(columns=constant_cols)
    print(f'Dropped {len(constant_cols)} constant columns')
    return df

def drop_high_missing_cols(df, threshold=0.70):
    cols_before = df.shape[1]
    null_pct = df.isnull().mean()
    high_missing = null_pct[null_pct > threshold].index.tolist()
    high_missing = [c for c in high_missing if c not in COMMON_FEATURES and c != TARGET]
    df = df.drop(columns=high_missing)
    print(f'Dropped {len(high_missing)} columns with >{threshold*100:.0f}% missing')
    return df

def parse_date_features(df):
    if DATE_COL in df.columns:
        dates = pd.to_datetime(df[DATE_COL], dayfirst=True, errors='coerce')
        df[f'{DATE_COL}_year'] = dates.dt.year
        df[f'{DATE_COL}_month'] = dates.dt.month
        df[f'{DATE_COL}_day'] = dates.dt.day
        df = df.drop(columns=[DATE_COL])
        print(f'Parsed {DATE_COL} into year/month/day features')

    if MONTH_COL in df.columns:
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
            'Jan25': 1, 'Feb25': 2, 'Mar25': 3, 'Apr25': 4, 'May25': 5,
            'Jun25': 6, 'Jul25': 7, 'Aug25': 8, 'Sep25': 9, 'Oct25': 10,
            'Nov25': 11, 'Dec25': 12,
        }
        df[f'{MONTH_COL}_num'] = df[MONTH_COL].map(month_map).fillna(0).astype(int)
        df = df.drop(columns=[MONTH_COL])
        print(f'Parsed {MONTH_COL} into numeric month')

    if 'F3889' in df.columns:
        f3889_type = df['F3889'].str.extract(r'^([A-Za-z]+)', expand=False)
        f3889_type_le = f3889_type.map({'G': 0, 'L': 1}).fillna(0).astype(int)
        df['F3889_type'] = f3889_type_le
        df['F3889_days'] = df['F3889'].str.extract(r'(\d+)', expand=False).astype(float)
        df = df.drop(columns=['F3889'])
        print('Parsed F3889 into type (label-encoded) and days features')

    return df

def encode_categoricals(df):
    encoders = {}
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna('MISSING')
        uniq = df[col].nunique()
        if uniq <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
            encoders[col] = 'onehot'
            print(f'One-hot encoded {col} ({uniq} categories)')
        else:
            target_mean = df.groupby(col)[TARGET].transform('mean')
            df[f'{col}_target_enc'] = target_mean
            df = df.drop(columns=[col])
            encoders[col] = 'target'
            print(f'Target-encoded {col} ({uniq} categories)')
    return df, encoders

def impute_missing(df):
    for col in df.columns:
        if col == TARGET:
            continue
        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'MISSING')
    print('Imputed missing values')
    return df

def select_features(X, y):
    common_present = [c for c in COMMON_FEATURES if c in X.columns]
    print(f'Common features present: {len(common_present)}')
    other_cols = [c for c in X.columns if c not in common_present]
    if len(other_cols) == 0:
        print('No additional features to select — using only common features')
        return common_present
    X_other = X[other_cols].select_dtypes(include=[np.number])
    valid_cols = X_other.columns[X_other.isnull().sum() == 0].tolist()
    X_other = X_other[valid_cols]
    n_to_select = min(MAX_FEATURES - len(common_present), len(valid_cols))
    if n_to_select <= 0:
        print('Common features already meet MAX_FEATURES threshold')
        return common_present
    selector = SelectKBest(mutual_info_classif, k=n_to_select)
    selector.fit(X_other, y)
    selected_mask = selector.get_support()
    selected_other = [valid_cols[i] for i in range(len(valid_cols)) if selected_mask[i]]
    selected_features = common_present + selected_other
    mi_scores = sorted(
        zip(selected_other, selector.scores_[selected_mask]),
        key=lambda x: -x[1]
    )
    print(f'Selected {len(common_present)} common + {len(selected_other)} MI-based = {len(selected_features)} total')
    print('Top 10 MI features:')
    for name, score in mi_scores[:10]:
        print(f'  {name}: {score:.4f}')
    return selected_features

def main():
    df = load_data()
    df = drop_constant_cols(df)
    df = drop_high_missing_cols(df, threshold=0.70)
    df = parse_date_features(df)
    df, encoders = encode_categoricals(df)
    df = impute_missing(df)

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    selected_features = select_features(X, y)
    X = X[selected_features]
    X_array = X.values
    y_array = y.values

    print(f'\nTraining data: {X_array.shape[0]} samples, {X_array.shape[1]} features')
    print(f'Target: {y_array.sum()} positive, {len(y_array) - y_array.sum()} negative')

    joblib.dump((X_array, y_array, selected_features, encoders),
                f'{OUTPUT_DIR}preprocessed_data.pkl')
    print(f'\nSaved preprocessing artifacts to {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
