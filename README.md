# 📈 Stock Price Forecasting using LSTM + MLflow

This project builds an **LSTM-based deep learning model** to forecast stock prices, with complete **MLflow** integration for tracking experiments, managing models, and visualizing training artifacts.

---

## ✅ Features

- 📊 Train LSTM models for time series forecasting
- 📈 Track parameters, metrics, artifacts (like prediction plots) using MLflow
- 🛡️ Automatically log and **register models** into MLflow Model Registry
- 🔥 Compare different model runs easily through **MLflow UI**
- 💬 Simple YAML-based configuration

---

## 🏗️ Project Structure

```
├── configs/
│   └── config.yaml             # Training configuration file
├── data/
│   └── raw/                    # Raw stock data CSVs (train/test)
├── outputs/
│   └── plots/                  # Prediction plots
├── scripts/
│   └── run_pipeline.py         # Main pipeline script
├── src/
│   ├── data/
│   │   ├── data_loader.py       # Load datasets
│   │   └── data_preprocessor.py # Preprocess datasets
│   ├── models/
│   │   └── lstm_model.py        # Define LSTM model
│   ├── trainer/
│   │   └── train_and_predict.py # Training and prediction logic
│   └── utils/
│       ├── logger.py            # Logger setup
│       └── plotter.py           # Plotting utilities
├── mlruns/                     # MLflow tracking data
├── requirements.txt            # Project dependencies
├── README.md
└── .gitignore
```

---

## 🚀 Quickstart Guide

Follow these steps to set up and run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock-price-forecasting-lstm.git
cd stock-price-forecasting-lstm
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
# Activate the environment
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Your Data

- Place your stock price **train.csv** and **test.csv** inside the `data/raw/` folder.
- Ensure your CSV files include required columns (e.g., **Open**, **Close**, **High**, **Low**).

Example structure:

```
Date,Open,High,Low,Close,Volume
2022-01-01,100,105,98,102,1000000
...
```

- Edit the configuration file `configs/config.yaml` to match your data paths and training settings:

```yaml
train_data_path: "data/Google_Stock_Price_Train.csv"
test_data_path: "data/Google_Stock_Price_Test.csv"
epochs: 100
batch_size: 32
timesteps: 60
feature_range: [0, 1]
```

### 5. Run the Full Training Pipeline

```bash
python scripts/run_pipeline.py
```

✅ This will:

- Train the LSTM model
- Log parameters, metrics, prediction plots into MLflow
- Automatically register the model in MLflow Model Registry
- Save output plots inside `outputs/plots/`

---

## 🧐 Monitor and Compare Runs with MLflow UI

Launch MLflow tracking server locally:

```bash
mlflow ui
```

Then open your browser at:

📈 [http://127.0.0.1:5000](http://127.0.0.1:5000)

Inside MLflow UI, you can:

- 📈 Track and compare runs
- 🔥 Visualize training and validation losses
- 🔼️ View generated prediction plots
- 🏋️️ Manage model versions (Staging, Production)

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`:

```plaintext
tensorflow==2.15.0
mlflow==2.10.2
pandas==2.2.1
numpy==1.26.4
matplotlib==3.8.4
scikit-learn==1.4.2
pyyaml==6.0.1
h5py==3.10.0
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 📊 Model Management

Each time you run training:

- The trained model is logged inside the `mlruns/` directory.
- It is **automatically registered** under the name `StockPriceLSTMModel`.
- You can easily **promote models** to **Staging** or **Production** from the MLflow UI.


---

## 🙌 Contributing

Contributions, issues, and feature requests are welcome!\
Feel free to ⭐ the repo if you find it useful!

---

## 📬 Contact

Created by Venkata Sai Rohith Andanala – feel free to reach out via GitHub Issues or pull requests.

---

