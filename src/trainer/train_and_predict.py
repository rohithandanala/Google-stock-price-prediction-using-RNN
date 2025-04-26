import os
import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.keras

from src.data.data_loader import load_data
from src.data.data_preprocessor import DataPreprocessor
from src.models.lstm_model import build_lstm
from src.utils.logger import setup_logger
from src.utils.plotter import plot_predictions

logger = setup_logger("Trainer")

def train_and_predict(config_path: str = "configs/config.yaml"):
    # Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # MLflow Setup
    mlflow.set_tracking_uri("mlruns")  # Use local directory for tracking
    mlflow.set_experiment(config.get("experiment_name", "Stock_Price_Forecasting"))

    with mlflow.start_run(run_name=config.get("run_name", "LSTM_Run")):
        run_id = mlflow.active_run().info.run_id

        # Log config parameters
        mlflow.log_params({
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "timesteps": config["timesteps"],
            "feature_range": config["feature_range"],
        })

        # Load Data
        train_df = load_data(config["train_data_path"])
        test_df = load_data(config["test_data_path"])

        training_set = train_df.iloc[:, 1:2].values
        test_set = test_df.iloc[:, 1:2].values

        # Preprocessing
        preprocessor = DataPreprocessor(feature_range=tuple(config["feature_range"]))
        training_scaled = preprocessor.fit_transform(training_set)
        X_train, y_train = preprocessor.create_sequences(training_scaled, config["timesteps"])

        # Build Model
        model = build_lstm(input_shape=(X_train.shape[1], 1))

        # Train Model
        history = model.fit(
            X_train, y_train,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            verbose=1
        )

        # Log final loss
        final_loss = history.history['loss'][-1]
        mlflow.log_metric("final_training_loss", final_loss)

        # Prepare Test Data
        total_data = pd.concat((train_df['Open'], test_df['Open']), axis=0)
        inputs = total_data[len(total_data) - len(test_df) - config["timesteps"]:].values
        inputs = inputs.reshape(-1, 1)
        inputs_scaled = preprocessor.transform(inputs)

        X_test = []
        for i in range(config["timesteps"], config["timesteps"] + len(test_set)):
            X_test.append(inputs_scaled[i - config["timesteps"]:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Predict
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = preprocessor.inverse_transform(predicted_stock_price)

        # Save Model to MLflow Registry
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name=config.get("registered_model_name", "StockPriceLSTMModel")
        )

        # Save Prediction Plot
        plot_dir = f"outputs/plots/{run_id}"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "prediction_plot.png")
        
        # Save plot locally
        plot_predictions(test_set, predicted_stock_price, save_path=plot_path)

        # Log Plot to MLflow
        mlflow.log_artifact(plot_path, artifact_path="plots")

    logger.info("Training, Prediction, and MLflow Logging completed successfully.")
