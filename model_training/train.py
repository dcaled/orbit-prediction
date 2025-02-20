# Now that the historical data is ready, we can move to the modelling phase. In this part, you are tasked with training
# and evaluating a model that predicts the future positions of an object, given as input the current position. You can
# tailor the input features as you see fit. For example, you may require as input more than one previous position, the
# velocity vector, or other external features from other data sources. Because this assignment is meant as a basis for
# discussion, we recommend you stick to the analysis of a single model.

import os
import sys
sys.path.insert(0, "..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import numpy as np
import pandas as pd
import optuna
from typing import Tuple, Dict, Any
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from setup_logger import logger
from utils import load_config

optuna.logging.set_verbosity(optuna.logging.WARNING)


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
        self.n_trials = self.config["models"]["n_trials"]
        self.param_ranges = self.config["models"]["hyperparameters"]
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


    def process_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates engineered features for model training.

        Args:
            df (pd.DataFrame): Raw orbit data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Feature matrix (X) and target variables (y).
        """
        # Sort data by epoch to ensure temporal order
        df = df.sort_values("epoch")
        epoch_min = df["epoch"].min()

        logger.info("Saving the first training position...")
        self.save_first_position(df[df["epoch"] == epoch_min])

        # Create delta as the difference between the record time and the initial time.
        df["delta"] = (df["epoch"] - epoch_min).astype(float)

        # Shift positions and velocities to get the last known values
        shift_features = ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"]
        for feature in shift_features:
            df[f"last_{feature}"] = df[feature].shift(1)

        # Drop rows with NaN values after shifting
        df.dropna(inplace=True)

        X = df[["delta"] + [f"last_{f}" for f in shift_features]]
        y = df[["pos_x", "pos_y", "pos_z"]]
        logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")

        return X, y


    @staticmethod
    def create_temporal_splits(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the data temporally into training (80%) and validation (20%) sets.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.DataFrame): Target values.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train-test split.
        """

        # Temporal split: 80% training, 20% validation
        split_index = int(0.8 * len(X))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        logger.info(f"Training shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Validation shape: X={X_val.shape}, y={y_val.shape}")
        return X_train, X_val, y_train, y_val


    def objective(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame,
                  y_val: np.ndarray) -> float:
        """
        Objective function for hyperparameter optimization with Optuna.

        Returns:
            float: Mean Squared Error (MSE) on validation set.
        """
        # Define the objective function for Optuna
        param = {
            "n_estimators": trial.suggest_int("n_estimators",
                                              self.param_ranges["n_estimators"]["min"],
                                              self.param_ranges["n_estimators"]["max"]),
            "learning_rate": trial.suggest_float("learning_rate",
                                                 self.param_ranges["learning_rate"]["min"],
                                                 self.param_ranges["learning_rate"]["max"]),
            "max_depth": trial.suggest_int("max_depth",
                                           self.param_ranges["max_depth"]["min"],
                                           self.param_ranges["max_depth"]["max"]),
            "subsample": trial.suggest_float("subsample",
                                             self.param_ranges["subsample"]["min"],
                                             self.param_ranges["subsample"]["max"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree",
                                                    self.param_ranges["colsample_bytree"]["min"],
                                                    self.param_ranges["colsample_bytree"]["max"])
        }

        model = XGBRegressor(objective="reg:squarederror", **param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        return mean_squared_error(y_val, model.predict(X_val))


    def optimize_model_parameters(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame,
                                  y_val: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """
        Optimizes hyperparameters using Optuna.

        Tuple[Dict[str, Any], float]:
            - Best hyperparameters found.
            - Corresponding best objective function value.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=self.n_trials)
        return study.best_params, study.best_value


    def train_final_model(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                          best_params: Dict[str, Any]) -> XGBRegressor:
        """
        Trains the final XGBoost model with the best hyperparameters.

        Returns:
            XGBRegressor: Trained model.
        """
        X_train_full = pd.concat([X_train, X_val], ignore_index=True)
        y_train_full = np.concatenate([y_train, y_val])

        final_model = XGBRegressor(objective="reg:squarederror", **best_params)
        final_model.fit(X_train_full, y_train_full)
        return final_model


    def save_first_position(self, first_position):
        """
        Saves the first position to a specified file, overwriting it if it exists.

        Args:
            first_position (pd.DataFrame): Dataframe containing the first known position of the space-object.
        """
        first_position.to_json(self.path_first_position, orient="records", lines=True)
        logger.info(f"First training position saved to {self.path_first_position}")


    def run_pipeline(self):
        """
        Runs the full data pipeline: loading, processing, training, and saving models.
        """
        logger.info("Loading data from storage...")
        df = self.load_orbits_from_folder()
        if df.empty:
            logger.error("No data found. Exiting pipeline.")
            return

        logger.info("Creating features...")
        X, y = self.process_data(df)

        logger.info("Creating temporal splits...")
        X_train, X_val, y_train, y_val = self.create_temporal_splits(X, y)

        logger.info("Optimizing and training models...")
        for axis in ["pos_x", "pos_y", "pos_z"]:
            logger.info(f"Optimizing model for {axis}...")
            best_params, best_value = self.optimize_model_parameters(X_train, y_train[axis], X_val, y_val[axis])
            logger.info(f"Best hyperparameters for {axis}: {best_params}. MSE: {best_value}")

            logger.info(f"Training final model for {axis} on the full dataset.")
            final_model = self.train_final_model(X_train, y_train[axis], X_val, y_val[axis], best_params)
            joblib.dump(final_model, os.path.join(self.path_models, f"model_{axis}.pkl"))
            logger.info(f"Final model for {axis} saved successfully.")


if __name__ == "__main__":
    model = OrbitPredictionModel("config.yaml")
    model.run_pipeline()
