<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=venom&height=200&color=0:0d1117,50:1a237e,100:0d1117&text=GHOST&section=header&reversal=false&textBg=false&fontSize=100&animation=twinkling&fontColor=ffffff"/>
</p>

<h3 align="center">AI-Powered Mule Account Detection System</h3>

<p align="center">
  <b>G</b>uarding <b>H</b>ostile <b>O</b>buscated <b>S</b>uspicious <b>T</b>ransactions
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://xgboost.readthedocs.io/"><img src="https://img.shields.io/badge/XGBoost-2024-orange?logo=xgboost" alt="XGBoost"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://shap.readthedocs.io/"><img src="https://img.shields.io/badge/SHAP-0.45-green" alt="SHAP"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License"></a>
  <a href="https://ghost-squadron-submittion.streamlit.app/"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App"></a>
</p>

---

##   Quick Stats

<p align="center">
  <img src="https://img.shields.io/badge/📊_Recall-97%25-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/🎯_Precision-94%25-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/📈_PR--AUC-0.975-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/🧠_Features-100-important?style=for-the-badge">
  <img src="https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Coud-important?style=for-the-badge">
</p>

---

##   The Problem

Banks face a growing wave of **cyber-enabled financial fraud** involving **mule accounts**. Criminals open or recruit accounts to receive, layer, and conceal stolen funds. Traditional rule-based systems can't keep up with evolving fraud patterns.

**GHOST solves this** by learning behavioral patterns from financial transaction data and detecting mule accounts with **97% recall** — before the money moves out.

---

##   How GHOST Works

```
                          ┌─────────────┐
                          │  Transaction │
                          │     Data     │
                          └──────┬──────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Preprocessing Engine   │
                    │  • Drop noise/constants  │
                    │  • Parse dates & tenors  │
                    │  • Encode categoricals   │
                    │  • Impute missing vals   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Feature Selection       │
                    │  • 18 domain features    │
                    │  • 82 MI-selected        │
                    │  = 100 total features    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  XGBoost Classifier      │
                    │  • Grid-searched params  │
                    │  • Scale pos weight=111  │
                    │  • 3-fold CV thresholding│
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │  Risk Score   │  │  SHAP Expl.  │  │  Dashboard   │
      │  0 – 100      │  │  per account │  │  Streamlit   │
      └──────────────┘  └──────────────┘  └──────────────┘
```

---

##   Live Demo

> ** [ghost-squadron-submittion.streamlit.app](https://ghost-squadron-submittion.streamlit.app/)**

Upload a CSV or test a single account — no installation needed.

<details>
<summary>   Click to preview the dashboard</summary>

| Batch Prediction | Model Insights |
|---|---|
| *Upload CSV, get risk scores + explanations* | *Confusion matrix, CV chart, SHAP features* |
| *(screenshot placeholder — capture from live app)* | *(screenshot placeholder)* |

</details>

---

##   Dataset

| Property | Value |
|---|---|
| Accounts | **9,082** (7,265 train / 1,817 test) |
| Suspicious | **81** (0.89% — extreme class imbalance) |
| Features | **3,923 anonymized** (F1 – F3924) |
| Target | **F3924** (1 = suspicious, 0 = legitimate) |
| Domain features | **18** provided by bank as commonly used signals |

---

##   Performance

### Held-Out Test Set

| Metric | Score |
|---|---|
| Recall | **1.000** (caught all 16 mules) |
| Precision | **1.000** (zero false alarms) |
| F2 Score | **1.000** |
| PR-AUC | **1.000** |

### 5-Fold Cross-Validation (Training Set)

| Fold | Recall | Precision | F2 | PR-AUC |
|---|---|---|---|---|
| 1 | 0.923 | 1.000 | 0.938 | 0.944 |
| 2 | 1.000 | 0.929 | 0.985 | 0.975 |
| 3 | 0.923 | 0.923 | 0.923 | 0.975 |
| 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5 | 1.000 | 0.867 | 0.970 | 0.982 |
| **Average** | **0.969 ± 0.038** | **0.944 ± 0.051** | **0.963 ± 0.029** | **0.975 ± 0.018** |

###   Top Predictive Signals

| Feature | Description | Importance |
|---|---|---|
| `F3912` | Anonymized transaction behavior signal | Highest |
| `F2230_num` | Month of transaction (parsed from date) | 2nd |
| `F3898` | Anonymized risk signal | 3rd |
| `F2030` | Anonymized pattern indicator | 4th |
| `F2956` | Anonymized behavioral feature | 5th |

---

##   Quick Start

```bash
# Clone
git clone https://github.com/Prateek-squadron/GHOST-SQUADRON-BOI.git
cd GHOST-SQUADRON-BOI

# Set up environment
python3 -m venv venv
source venv/bin/activate           # bash/zsh
# source venv/bin/activate.fish    # fish shell
pip install -r requirements.txt

# Place train.csv in project root, then:
python3 2_train.py                 # full pipeline: preprocess → train → evaluate

# Run inference on new data
python3 3_predict.py --batch ./data.csv

# Launch dashboard
streamlit run app.py
```

---

##   Project Structure

```
├── 2_train.py              # 🧠 Training pipeline
├── 3_predict.py            # 🔮 Inference script
├── app.py                  #   Streamlit dashboard
├── notebook.ipynb          # 📓 Jupyter notebook (analysis + plots)
├── requirements.txt        # 📦 Dependencies
├── model/                  # 💾 Trained artifacts
│   ├── xgb_model.json
│   ├── shap_explainer.pkl
│   ├── threshold.pkl
│   ├── selected_features.pkl
│   ├── test_eval.pkl
│   └── shap_sample.pkl
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🛠️ Built With

| Tool | Purpose |
|---|---|
| [Python](https://python.org) | Core language |
| [XGBoost](https://xgboost.readthedocs.io/) | Gradient boosting classifier |
| [SHAP](https://shap.readthedocs.io/) | Explainability & feature importance |
| [Scikit-learn](https://scikit-learn.org/) | Preprocessing, metrics, CV |
| [Streamlit](https://streamlit.io/) | Interactive dashboard |
| [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) | Visualizations |

---

<p align="center">
  <a href="https://ghost-squadron-submittion.streamlit.app/">  Live Demo</a>
  ·
  <a href="https://github.com/Prateek-squadron">👤 Author</a>
  ·
  <a href="LICENSE">📝 MIT License</a>
</p>
