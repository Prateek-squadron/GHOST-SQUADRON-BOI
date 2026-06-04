# GHOST SQUADRON BOI — Mule Account Detection System

AI/ML-powered classification system to detect suspicious mule accounts from financial transaction data using XGBoost with SHAP explainability.

## Problem

Banks face growing cyber-enabled financial fraud involving mule accounts used to receive, transfer, and conceal fraudulent funds. Traditional rule-based systems fail to catch evolving fraud patterns. This system achieves **~97% recall** and **~94% precision** in cross-validation.

## Dataset

- **9,082 accounts** (81 suspicious / 9,001 legitimate — 0.89% positive rate)
- **3,923 anonymized features** (F1–F3924) + target F3924
- 18 domain-provided common features prioritized for selection
- Train/test split: **80/20 stratified** (test set held out for evaluation)

## Pipeline

```
1_preprocess.py   →  2_train.py   →  3_predict.py
                        ↓
                  notebook.ipynb  (analysis + plots)
                  app.py          (Streamlit dashboard)
```

| Step | Script | Description |
|---|---|---|
| Preprocessing | `1_preprocess.py` | Drop constants, impute missing, parse dates, encode categoricals, MI-based feature selection (100 features) |
| Training | `2_train.py` | XGBoost with grid search, class weighting, 3-fold CV, F2-optimal threshold tuning, SHAP explainer |
| Inference | `3_predict.py` | Batch CSV prediction or single-row input, outputs risk score + SHAP explanations |
| Notebook | `notebook.ipynb` | Full walkthrough with EDA, visualizations, evaluation, SHAP analysis |
| Dashboard | `app.py` | Streamlit web app with CSV upload, single predict, and model insights |

## Results

| Metric | 5-Fold CV (avg ± std) | Held-Out Test |
|---|---|---|
| Recall | **0.969 ± 0.038** | **1.000** |
| Precision | **0.944 ± 0.051** | **1.000** |
| F2 Score | **0.963 ± 0.029** | **1.000** |
| PR-AUC | **0.975 ± 0.018** | **1.000** |

Top features by SHAP importance: `F3912`, `F2230_num` (month), `F3898`, `F2030`, `F2956`.

## Setup

```bash
git clone https://github.com/Prateek-squadron/GHOST-SQUADRON-BOI.git
cd GHOST-SQUADRON-BOI
python3 -m venv venv
source venv/bin/activate       # bash/zsh
# source venv/bin/activate.fish  # fish shell
pip install -r requirements.txt
```

## Usage

### Train the model
Place `train.csv` in the project root, then:
```bash
python3 1_preprocess.py
python3 2_train.py
```

### Run inference
```bash
# Batch prediction on CSV
python3 3_predict.py --batch ./path/to/data.csv

# Interactive single prediction
python3 3_predict.py
```

### Launch dashboard
```bash
streamlit run app.py
```

### Run notebook
```bash
jupyter notebook notebook.ipynb
```

## Project Structure

```
├── 1_preprocess.py       # Data cleaning & feature engineering
├── 2_train.py            # XGBoost training with grid search + SHAP
├── 3_predict.py          # Inference script (batch + single)
├── app.py                # Streamlit dashboard
├── notebook.ipynb        # Jupyter notebook with full analysis
├── requirements.txt      # Dependencies
├── model/                # Trained model artifacts
│   ├── xgb_model.json
│   ├── threshold.pkl
│   ├── selected_features.pkl
│   └── shap_explainer.pkl
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT
