import json
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from setup_logger import logger
from utils import load_config
from sklearn.linear_model import LinearRegression


class OrbitInference:
    def __init__(self, config_path="../config.yaml"):
        """
        Initializes the OrbitInference class by loading the configuration and trained models.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.config = load_config(config_path)
        # Interval between predictions (Time step in seconds)
        self.delta = self.config["inference"]["delta"]
        # Number of future predictions.
        self.n_predictions = self.config["inference"]["number_of_predictions"]

        self.path_models = self.config["models"]["path_models"]
        self.models = self.load_models()
        # Initial position data
        self.path_first_position = self.config["models"]["path_first_position"]
        self.first_position = self.load_first_position()

    def load_models(self) -> Dict[str, LinearRegression]:
        """
        Loads trained models from disk.

        Returns:
            Dict[str, LinearRegression]: Dictionary containing models for each target variable.
        """
        models = {}
        for axis in ["pos_x", "pos_y", "pos_z"]:
            model_path = os.path.join(self.path_models, f"{axis}_model.pkl")
            if os.path.exists(model_path):
                logger.info(f"Loading model for {axis}...")
                models[axis] = joblib.load(model_path)
            else:
                raise FileNotFoundError(f"Model for {axis} not found. Train the model first.")
        return models

    def load_first_position(self) -> Any | None:
        """
        Loads the first position from a specified file.
        """
        if os.path.exists(self.path_first_position):
            with open(self.path_first_position, "r") as f:
                return json.load(f)

    def predict(self, x_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predicts outputs for each model.

        Args:
            x_test (pd.DataFrame): Test feature matrix.

        Returns:
            Dict[str, np.ndarray]: Predictions for each target variable.
        """
        return {col: self.models[col].predict(x_test) for col in self.models}

    def run_inference(self, last_position: Dict[str, float]) -> List[List[float]]:
        """
        Runs the inference pipeline to predict future positions.

        Args:
            last_position (Dict[str, float]): Last known position data.

        Returns:
            List[List[float]]: List of predicted positions with timestamps.
        """
        future_predictions = []

        epoch_last = last_position["epoch"]
        pos_x, pos_y, pos_z = last_position["pos_x"], last_position["pos_y"], last_position["pos_z"]
        vel_x, vel_y, vel_z = last_position["vel_x"], last_position["vel_y"], last_position["vel_z"]
        t_min = self.first_position["epoch"]

        for i in range(self.n_predictions):
            pred_time = epoch_last + (i * self.delta * 1000)
            delta_time = (pred_time - t_min)

            X_pred = pd.DataFrame([[pred_time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, delta_time]],
                                  columns=["epoch", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "delta_time"])

            prediction = self.predict(X_pred)
            pos_x_pred, pos_y_pred, pos_z_pred = prediction["pos_x"][0], prediction["pos_y"][0], prediction["pos_z"][0]

            future_predictions+=[{
                "epoch": pred_time,
                "x": pos_x_pred,
                "y": pos_y_pred,
                "z": pos_z_pred
            }]

            pos_x, pos_y, pos_z = pos_x_pred, pos_y_pred, pos_z_pred

        return future_predictions


if __name__ == "__main__":
    inference = OrbitInference()

    last_position = {
        "epoch": 1739307420000,
        "pos_x": -2.787127,
        "pos_y": 5994.151251,
        "pos_z": 3726.167103,
        "vel_x": 17226.162,
        "vel_y": -39257.044,
        "vel_z": 62793.22
    }

    predictions = inference.run_inference(last_position)
    print("Future Predictions:")
    for pred in predictions:
        print(pred)
