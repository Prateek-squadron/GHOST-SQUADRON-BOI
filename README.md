# GHOST SQUADRON BOI — Mule Account Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ghost-squadron-submittion.streamlit.app/)

AI/ML-powered classification system to detect suspicious mule accounts from financial transaction data using XGBoost with SHAP explainability.

## Problem

Banks face growing cyber-enabled financial fraud involving mule accounts used to receive, transfer, and conceal fraudulent funds. Traditional rule-based systems fail to catch evolving fraud patterns.

## Dataset

- **9,082 accounts** (81 suspicious / 9,001 legitimate — 0.89% positive rate)
- **3,923 anonymized features** (F1–F3924) + target `F3924`
- **18 domain-provided common features** prioritized for selection
- **80/20 stratified split** — test set held out for final evaluation only

## Pipeline

```
train.csv  ──►  2_train.py  ──►  model/  ──►  3_predict.py
                                              app.py (Streamlit)
                                              notebook.ipynb
```

| Step | File | Description |
|---|---|---|
| Training | `2_train.py` | Preprocessing + feature selection (100 features) + XGBoost grid search + SHAP explainer + test evaluation |
| Inference | `3_predict.py` | Batch CSV prediction or single-row input; outputs risk score + SHAP explanations |
| Dashboard | `app.py` | Streamlit web app with CSV upload, single predict, and model insights |
| Notebook | `notebook.ipynb` | Full walkthrough with EDA, visualizations, evaluation, SHAP analysis |

## Performance

| Metric | 5-Fold CV (avg ± std) | Held-Out Test |
|---|---|---|
| Recall | **0.969 ± 0.038** | **1.000** |
| Precision | **0.944 ± 0.051** | **1.000** |
| F2 Score | **0.963 ± 0.029** | **1.000** |
| PR-AUC | **0.975 ± 0.018** | **1.000** |

**Top SHAP features:** `F3912`, `F2230_num` (month), `F3898`, `F2030`, `F2956`

## Dashboard

The Streamlit dashboard (`app.py`) includes three tabs:

###  Batch Predict
Upload a CSV → get risk scores, predictions, downloadable results, risk score histogram, and per-account SHAP waterfall explanations.

###  Single Predict
Enter feature values manually → instant risk score and top contributing features.

###  Model Insights
Confusion matrix, precision-recall curve, SHAP summary plots, risk score distribution, and cross-validation metrics — all rendered live.

## Setup

```bash
git clone https://github.com/Prateek-squadron/GHOST-SQUADRON-BOI.git
cd GHOST-SQUADRON-BOI
python3 -m venv venv
source venv/bin/activate              # bash/zsh
# source venv/bin/activate.fish       # fish shell
pip install -r requirements.txt
```

## Usage

### Train the model
Place `train.csv` in the project root, then:
```bash
python3 2_train.py
```
This runs preprocessing, feature selection, grid search, CV threshold tuning, SHAP computation, and test set evaluation in one step.

### Run inference
```bash
# Batch prediction on CSV
python3 3_predict.py --batch ./path/to/data.csv

# Interactive single prediction
python3 3_predict.py
```

### Launch dashboard locally
```bash
streamlit run app.py
```

### Run notebook
```bash
jupyter notebook notebook.ipynb
```

### Or use the live app
[https://ghost-squadron-submittion.streamlit.app/](https://ghost-squadron-submittion.streamlit.app/)

## Project Structure

```
├── 2_train.py              # Training pipeline (preprocess → train → evaluate)
├── 3_predict.py            # Inference script (batch + single)
├── app.py                  # Streamlit dashboard with graphs
├── notebook.ipynb          # Jupyter notebook with full analysis
├── requirements.txt        # Dependencies
├── model/                  # Trained model artifacts
│   ├── xgb_model.json      # XGBoost model
│   ├── threshold.pkl       # Optimal F2 threshold
│   ├── selected_features.pkl
│   ├── shap_explainer.pkl  # SHAP TreeExplainer
│   └── test_eval.pkl       # Test set predictions + metrics
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT
