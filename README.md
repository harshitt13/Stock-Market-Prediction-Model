# Hybrid Stock Market Prediction Model

<div align="center">

![GitHub last commit](https://img.shields.io/github/last-commit/harshitt13/Stock-Market-Prediction-Model)
![GitHub repo size](https://img.shields.io/github/repo-size/harshitt13/Stock-Market-Prediction-Model)
![GitHub stars](https://img.shields.io/github/stars/harshitt13/Stock-Market-Prediction-Model)
[![wakatime](https://wakatime.com/badge/github/harshitt13/Stock-Market-Prediction-Using-ML.svg)](https://wakatime.com/badge/github/harshitt13/Stock-Market-Prediction-Using-ML)

</div>

## 🎯 Project Overview

This project implements an advanced stock price prediction model utilizing a **Stacking Meta-Ensemble**. The system combines two sophisticated machine learning approaches:
1. **Tree Ensemble (XGBoost + Random Forest)** 
2. **Deep Bidirectional LSTM (PyTorch)**

A Ridge Regression meta-learner evaluates and optimally weights the predictions of these two models to generate highly accurate future price predictions with **95% confidence intervals**.

Video Demonstration - https://youtu.be/z8sXhWrwU0o

## 💻 Features

- **Automated Data Processing:** Retrieves data from Yahoo Finance (`yfinance`) and generates 27 engineered features including MACD, Bollinger Bands, ATR, OBV, and lagged components.
- **Tree Ensemble:** Uses a `VotingRegressor` combining tuned XGBoost and Random Forest.
- **Bidirectional LSTM:** Deep 3-layer architecture with Batch Normalization, Dropout, Huber Loss, and learning rate scheduling in PyTorch.
- **Stacking Meta-Ensemble:** Ridge regression dynamically learns the optimal combination weights of the underlying sub-models.
- **Comprehensive Evaluation:** Built-in module scoring RMSE, MAE, MAPE, R², and Directional Accuracy.
- **Visualization Dashboard:** Automatically generates professional, dark-themed charts containing actual vs predicted data, residual histograms, future forecasts, and feature importance.
- **CLI Interface:** Dynamic command-line inputs for custom tickers, date ranges, and forecast windows.

## 📚 Prerequisites

- Python 3.x
- PyTorch (`torch`)
- XGBoost
- scikit-learn
- yfinance
- pandas
- numpy
- matplotlib
- joblib

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/harshitt13/Stock-Market-Prediction-Model.git
cd Stock-Market-Prediction-Model
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
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
│   ├── tree_model.py        # XGBoost + RF ensemble implementation
│   ├── lstm_model.py        # Deep BiLSTM implementation (PyTorch)
│   ├── evaluate.py          # Metrics module (MAPE, R², RMSE, DA)
│   ├── visualize.py         # Matplotlib dashboard generator
│   └── main.py              # CLI orchestrator & stacking meta-ensemble
├── models/                  # Saved .pkl and .pt model files
├── tests/                   # Pytest suite
├── images/                  # Generated performance and forecast visualizations
├── requirements.txt         
├── pytest.ini               
└── README.md                
```

## 🔧 Usage

The entire pipeline (fetching, training both sub-models, creating the meta-ensemble, and evaluation) is executed via the CLI.

```bash
python src/main.py --ticker AAPL --start 2015-01-01 --days 30
```

### Command Line Arguments
- `--ticker`: The stock symbol to predict (default: `AAPL`).
- `--start`: Historical data start date in `YYYY-MM-DD` (default: `2010-01-01`).
- `--days`: Number of future business days to project (default: `30`).

## ⚖️ Model Components

1. **Tree Model (`tree_model.py`)**
   - Combines XGBoost and Random Forest.
   - Learns non-linear feature interactions and isolates the most important technical indicators.
   - Outputs feature importance charts.

2. **Bidirectional LSTM (`lstm_model.py`)**
   - Deep neural network capturing complex temporal dependencies over a 90-day lookback window.
   - Robust to outliers using Huber Loss.

3. **Hybrid Meta-Ensemble (`main.py`)**
   - Uses Ridge Regression on the sub-models' validation outputs.
   - Replaces naive 50/50 averaging with mathematically optimal weighting.

## 📏 Performance Metrics

Models are evaluated via:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²) Score
- Directional Accuracy (%)

Comparison tables are saved to `data/model_comparison.csv` and visualized in the `images/` directory.

## 🧪 Testing

Run the test suite via `pytest` to verify the components are functioning properly:
```bash
python -m pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Limitations

- Stock predictions are inherently probabilistic. This model uses only technical indicators, which omits fundamental (P/E, earnings) and sentiment (news, macroeconomics) data.
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
