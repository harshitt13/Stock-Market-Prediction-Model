# Hybrid Stock Market Prediction Model

<div align="center">

![GitHub last commit](https://img.shields.io/github/last-commit/harshitt13/Stock-Market-Prediction-Model)
![GitHub repo size](https://img.shields.io/github/repo-size/harshitt13/Stock-Market-Prediction-Model)
![GitHub stars](https://img.shields.io/github/stars/harshitt13/Stock-Market-Prediction-Model)
[![wakatime](https://wakatime.com/badge/github/harshitt13/Stock-Market-Prediction-Using-ML.svg)](https://wakatime.com/badge/github/harshitt13/Stock-Market-Prediction-Using-ML)

</div>

## 🎯 Project Overview

This project implements an advanced stock price prediction model utilizing a **Stacking Meta-Ensemble** with **walk-forward validation** and **classification-primary evaluation**. The system combines three sophisticated machine learning approaches:

1. **Tree Ensemble (XGBoost + Random Forest)**
2. **Deep Bidirectional LSTM (PyTorch)**
3. **Time-Series Transformer (PyTorch)**

A non-linear XGBoost meta-learner evaluates and optimally weights the predictions of these three models using **out-of-fold predictions** (leakage-free) to generate future price predictions with **Volatility-Adjusted 95% Confidence Intervals** (calibrated and validated).

**Core Research Question:** Does volatility-conditioned dynamic ensemble weighting (via the XGBoost meta-learner using VIX) improve predictive robustness over static ensemble weighting, particularly during high-volatility market regimes?

Video Demonstration - https://youtu.be/z8sXhWrwU0o

## 💻 Features

- **Walk-Forward Validation:** Expanding-window cross-validation ensures no future data leakage. Configurable train/test/step sizes via CLI.
- **Classification-Primary Evaluation:** Directional Accuracy (up/down), F1 Score, Precision, Recall, and Confusion Matrix are the primary metrics. Regression metrics (RMSE, MAE, MAPE, R²) are secondary.
- **Out-of-Fold Meta-Ensemble:** XGBoost meta-learner trained on out-of-fold predictions from the base models, eliminating in-sample leakage.
- **Macro-Economic Engine:** Enriches the target stock with S&P 500 (`^GSPC`), Volatility Index (`^VIX`), and Treasury Yield (`^TNX`).
- **Time-Series Transformer:** Deep architecture using `nn.TransformerEncoder` with Positional Encoding.
- **Tree Ensemble:** `VotingRegressor` combining tuned XGBoost and Random Forest.
- **Bidirectional LSTM:** Deep 3-layer architecture with Batch Normalization, Dropout, Huber Loss.
- **Bayesian Optimization (Optuna):** Safe hyperparameter search — never touches the outer test fold.
- **Baseline Comparison:** Naive (zero-change) and ARIMA baselines included in all comparison tables.
- **Confidence Interval Calibration:** Empirical coverage validated against the nominal 95% level with explicit reporting.
- **Visualization Dashboard:** Dark-themed charts: actual vs predicted, residuals, future forecasts, confusion matrices, walk-forward fold boundaries, and feature importance.
- **API-Ready Architecture:** Core pipeline logic in clean, importable functions (`run_pipeline()`) — ready for FastAPI wrapping.

## 📚 Prerequisites

- Python 3.10+
- PyTorch (`torch`)
- XGBoost
- scikit-learn
- yfinance
- pandas, numpy, matplotlib
- statsmodels (ARIMA baseline)
- seaborn (confusion matrix visualization)
- scipy (confidence interval z-scores)
- Optuna (hyperparameter optimization)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/harshitt13/Stock-Market-Prediction-Model.git
cd Stock-Market-Prediction-Model
```

2. Create a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```
Stock-Market-Prediction-Model/
│
├── data/                    # Raw, predicted CSVs and model comparison tables
├── src/
│   ├── fetch_data.py        # Data retrieval and 27-feature engineering
│   ├── tree_model.py        # XGBoost + RF ensemble (walk-forward)
│   ├── lstm_model.py        # Deep BiLSTM (PyTorch, walk-forward)
│   ├── transformer_model.py # Time-Series Transformer (walk-forward, Optuna-safe)
│   ├── evaluate.py          # Metrics (classification-primary, regression-secondary)
│   ├── baselines.py         # Naive + ARIMA baselines
│   ├── calibration.py       # Confidence interval calibration
│   ├── walk_forward.py      # Walk-forward validation engine
│   ├── visualize.py         # Dashboard generator (dark-themed)
│   └── main.py              # CLI orchestrator & importable pipeline
├── models/                  # Saved .pkl and .pt model files
├── tests/                   # Pytest suite (walk-forward, baselines, calibration, etc.)
├── images/                  # Generated visualizations
├── requirements.txt
├── pytest.ini
└── README.md
```

## 🔧 Usage

### Full Pipeline (CLI)

```bash
python src/main.py --ticker AAPL --start 2015-01-01 --days 30
```

### With Optuna Optimization

```bash
python src/main.py --ticker AAPL --start 2015-01-01 --days 30 --optimize
```

### Custom Walk-Forward Parameters

```bash
python src/main.py --ticker AAPL --start 2010-01-01 --days 30 \
    --min-train 504 --test-size 63 --step-size 63
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ticker` | `AAPL` | Stock ticker symbol |
| `--start` | `2010-01-01` | Historical data start date |
| `--days` | `30` | Number of future business days to predict |
| `--optimize` | `false` | Enable Optuna hyperparameter search (safe) |
| `--min-train` | `504` | Minimum training window (~2 years) |
| `--test-size` | `63` | Test fold size (~3 months) |
| `--step-size` | `63` | Step between folds |

### Programmatic Use (API-Ready)

```python
from main import run_pipeline

result = run_pipeline(
    ticker="AAPL",
    start_date="2015-01-01",
    future_days=30,
)

# Access results
print(result["comparison_df"])
print(result["calibration_report"])
```

## ⚖️ Model Components

1. **Time-Series Transformer (`transformer_model.py`)**
   - Multi-Head Attention for long-term trend capture.
   - Walk-forward validated per fold.

2. **Bidirectional LSTM (`lstm_model.py`)**
   - 3-layer BiLSTM capturing 90-day temporal dependencies.
   - Per-fold scaler fitting prevents data leakage.

3. **Tree Model (`tree_model.py`)**
   - XGBoost + Random Forest VotingRegressor.
   - Captures non-linear feature interactions.

4. **Hybrid Meta-Ensemble (`main.py`)**
   - XGBoost meta-learner on **out-of-fold** base model predictions.
   - VIX-conditioned dynamic weighting shifts trust between models based on market volatility.

5. **Baselines (`baselines.py`)**
   - **Naive:** Predicts tomorrow = today (zero-change sanity check).
   - **ARIMA(5,1,0):** Classical time-series baseline via statsmodels.

## 📏 Evaluation Methodology

### Walk-Forward Validation

All models are evaluated using expanding-window walk-forward cross-validation:
- Train on `[0, T]`, test on `[T, T+k]`, expand and repeat.
- No shuffling. No future data leakage.
- Metrics reported per-fold and aggregated.

### Primary Metrics (Classification / Directional)

| Metric | Description |
|--------|-------------|
| **Directional Accuracy** | % of days predicted direction matches actual |
| **F1 Score** | Harmonic mean of precision and recall (macro) |
| **Precision** | Correct directional predictions / total predicted |
| **Recall** | Correct directional predictions / total actual |
| **Confusion Matrix** | 2×2 matrix: correct/false up/down predictions |

### Secondary Metrics (Regression)

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| MAPE (%) | Mean Absolute Percentage Error |
| R² | Coefficient of Determination |

### Confidence Interval Validation

The 95% volatility-adjusted confidence interval is calibrated by checking what percentage of actual values fall within the predicted bounds. Coverage is reported explicitly — if far from 95%, it's flagged.

Comparison tables are saved to `data/model_comparison.csv` and visualized in `images/`.

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

Tests cover:
- Walk-forward splitter (no leakage, expanding windows, fold correctness)
- Classification metrics (directional accuracy, F1, confusion matrix)
- Baseline models (naive, ARIMA)
- Confidence interval calibration
- Tree/LSTM model training with walk-forward folds

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Limitations

- Stock predictions are inherently probabilistic. This model uses advanced technicals and macroeconomic contexts, but omits fundamental analysis (P/E, Debt/Equity) and NLP sentiment (news).
- Model performance depends heavily on structural market regimes.
- Past performance does not guarantee future results. **Do not use for real financial trading.**

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📫 Contact

<div align="center">

[![Github](https://img.shields.io/badge/-Github-000?style=flat&logo=Github&logoColor=white)](https://github.com/harshitt13)

**Harshit Kushwaha 🧑‍💻**
Developer

📧 find.harshitkushwaha@gmail.com

</div>

---
