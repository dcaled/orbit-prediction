# Now that the historical data is ready, we can move to the modelling phase. In this part, you are tasked with training
# and evaluating a model that predicts the future positions of an object, given as input the current position. You can
# tailor the input features as you see fit. For example, you may require as input more than one previous position, the
# velocity vector, or other external features from other data sources. Because this assignment is meant as a basis for
# discussion, we recommend you stick to the analysis of a single model.

import os
import sys
sys.path.insert(0, '..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from typing import Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from setup_logger import logger
from utils import load_config


class OrbitPredictionModel:
    def __init__(self, config_path: str):
        """
        Initializes the OrbitPredictionModel class with configuration parameters.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config = load_config(config_path)
        self.path_data = self.config["data"]["path_clean_data"]
        self.path_first_position = self.config["models"]["path_first_position"]
        self.path_models = self.config["models"]["path_models"]
        self.models = {}


    def load_orbits_from_folder(self) -> pd.DataFrame:
        """
        Reads all persisted orbit data from a given folder and loads it into a Pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing all loaded orbit data.
        """
        orbit_data = []
        for filename in os.listdir(self.path_data):
            if filename.endswith(".json"):
                with open(os.path.join(self.path_data, filename), "r") as f:
                    orbit_data+=[json.load(f)]

        if not orbit_data:
            logger.warning("No orbit data found.")
            return pd.DataFrame()

        return pd.json_normalize(orbit_data)


    @staticmethod
    def create_temporal_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data temporally into training (80%) and testing (20%) sets.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets.
        """
        df = df.sort_values("epoch")

        # Create delta_time as the difference between the record time and the initial time.
        epoch_min = df["epoch"].min()
        df["delta_time"] = df["epoch"] - epoch_min

        split_time = df["epoch"].quantile(0.8)
        train_data = df[df["epoch"] < split_time]
        test_data = df[df["epoch"] >= split_time]

        return train_data, test_data


    @staticmethod
    def create_training_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare the feature matrix X and target vectors y_pos_x, y_pos_y, y_pos_z.

        Returns:
            Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]: Feature matrix and target values.
        """

        # Drop the last row to align X and Y properly for prediction
        X = df.iloc[:-1, :]

        # Target variables are the next position (shifted by one)
        y_pos_x = df["pos_x"].shift(-1).dropna()
        y_pos_y = df["pos_y"].shift(-1).dropna()
        y_pos_z = df["pos_z"].shift(-1).dropna()

        return X, y_pos_x, y_pos_y, y_pos_z


    def fit_model(self, x_train: pd.DataFrame, y_train_pos_x: np.ndarray, y_train_pos_y: np.ndarray,
                  y_train_pos_z: np.ndarray):
        """
        Trains separate Linear Regression models with hyperparameter tuning for each target variable and saves them.

        Args:
            x_train (pd.DataFrame): Feature matrix.
            y_train_pos_x, y_train_pos_y, y_train_pos_z (np.ndarray): Target vectors for each position.
        """
        # Ensure model directory exists
        os.makedirs(self.path_models, exist_ok=True)

        # Fit and save model for each target variable
        self.models["pos_x"] = LinearRegression().fit(x_train, y_train_pos_x)
        joblib.dump(self.models["pos_x"], os.path.join(self.path_models, "model_pos_x.pkl"))

        self.models["pos_y"] = LinearRegression().fit(x_train, y_train_pos_y)
        joblib.dump(self.models["pos_y"], os.path.join(self.path_models, "model_pos_y.pkl"))

        self.models["pos_z"] = LinearRegression().fit(x_train, y_train_pos_z)
        joblib.dump(self.models["pos_z"], os.path.join(self.path_models, "model_pos_z.pkl"))


    def predict(self, x_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predicts outputs for each model.

        Args:
            x_test (pd.DataFrame): Test feature matrix.

        Returns:
            Dict[str, np.ndarray]: Predictions for each target variable.
        """
        return {col: self.models[col].predict(x_test) for col in self.models}

    def evaluate(self, x_test: pd.DataFrame, y_test: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluates models using Mean Squared Error.

        Args:
            x_test (pd.DataFrame): Test feature matrix.
            y_test (Dict[str, np.ndarray]): True values for the test set.

        Returns:
            Dict[str, float]: Mean Squared Error for each model.
        """
        y_pred = self.predict(x_test)
        return {col: mean_squared_error(y_test[col], y_pred[col]) for col in self.models}

    def save_first_position(self, df):
        """
        Saves the first position to a specified file, overwriting it if it exists.

        Args:
            df (pd.DataFrame): Feature matrix.
        """
        first_position = df[df["epoch"]==df["epoch"].min()]
        first_position.to_json(self.path_first_position, orient='records', lines=True)
        logger.info(f"First training position saved to {self.path_first_position}")

    def run_pipeline(self):
        """
        Executes the entire data processing and model training pipeline.
        """
        logger.info("Loading data from storage...")
        df = self.load_orbits_from_folder()

        logger.info("Creating temporal splits...")
        train_data, test_data = self.create_temporal_splits(df)
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info(f"Test data shape: {test_data.shape}")

        logger.info("Saving the first training position...")
        self.save_first_position(train_data)

        logger.info("Creating training and test sets...")
        x_train, y_train_pos_x, y_train_pos_y, y_train_pos_z = self.create_training_set(train_data)
        x_test, y_test_pos_x, y_test_pos_y, y_test_pos_z = self.create_training_set(test_data)

        logger.info("Training models...")
        self.fit_model(x_train, y_train_pos_x, y_train_pos_y, y_train_pos_z)

        logger.info("Evaluating models...")
        mse_scores = self.evaluate(x_test, {"pos_x": y_test_pos_x, "pos_y": y_test_pos_y, "pos_z": y_test_pos_z})
        logger.info(f"Mean Squared Errors: {mse_scores}")



if __name__ == "__main__":
    model = OrbitPredictionModel("config.yaml")
    model.run_pipeline()
