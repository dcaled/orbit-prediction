import os

import joblib

from utils import load_config

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Tuple, Dict

from setup_logger import logger


def create_training_set(path_data: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads dataset from the specified path and prepares training data.

    Args:
        path_data (str): Path to the dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Feature matrix (X) and target values (Y).
    """
    # Replace this with actual file reading logic
    np.random.seed(42)
    num_records = 1000
    time = np.arange(num_records)
    x = np.random.uniform(-10, 10, num_records)
    y = np.random.uniform(-10, 10, num_records)
    z = np.random.uniform(-10, 10, num_records)
    vx = np.random.uniform(-1, 1, num_records)
    vy = np.random.uniform(-1, 1, num_records)
    vz = np.random.uniform(-1, 1, num_records)

    df = pd.DataFrame({
        "epoch": time,
        "pos_x": x,
        "pos_y": y,
        "pos_z": z,
        "vel_x": vx,
        "vel_y": vy,
        "vel_z": vz,
    })

    # Drop last row to align X and Y properly
    X = df.iloc[:-1, 1:]  # Exclude "epoch"
    Y = df[["pos_x", "pos_y", "pos_z"]].shift(-1).dropna()

    return X, Y


def create_splits(X: pd.DataFrame, Y: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Splits data into training and testing sets.

    Args:
        X (pd.DataFrame): Feature matrix.
        Y (pd.DataFrame): Target variables.

    Returns:
        Tuple: Training and test sets for X and Y.
    """
    x_train, x_test = train_test_split(X, test_size=0.2, random_state=42)
    y_train = {}
    y_test = {}
    for col in Y.columns:
        y_train[col], y_test[col] = train_test_split(Y[col], test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test



def fit_model(x_train: pd.DataFrame, y_train: Dict[str, np.ndarray], path_models) -> Dict[str, LinearRegression]:
    """
    Trains a separate Linear Regression model for each target variable and saves them.

    Args:
        x_train (pd.DataFrame): Training feature matrix.
        y_train (Dict[str, np.ndarray]): Dictionary of training targets.

    Returns:
        Dict[str, LinearRegression]: Trained models for each target variable.
    """
    os.makedirs(path_models, exist_ok=True)  # Ensure model directory exists

    models = {}
    for col in y_train:
        model_path = os.path.join(path_models, f"{col}_model.pkl")

        if os.path.exists(model_path):
            logger.info(f"Loading existing model for {col}...")
            models[col] = joblib.load(model_path)
        else:
            logger.info(f"Training new model for {col}...")
            model = LinearRegression().fit(x_train, y_train[col])
            joblib.dump(model, model_path)  # Save model
            models[col] = model

    return models


def load_models(path_models) -> Dict[str, LinearRegression]:
    """
    Loads all saved models from disk.

    Returns:
        Dict[str, LinearRegression]: Dictionary of loaded models.
    """
    models = {}
    for axis in ["pos_x", "pos_y", "pos_z"]:
        model_path = os.path.join(path_models, f"{axis}_model.pkl")
        if os.path.exists(model_path):
            logger.info(f"Loading model for {axis}...")
            models[axis] = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model for {axis} not found. Train the model first.")

    return models

def predict(models: Dict[str, LinearRegression], x_test: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Predicts outputs for each model.

    Args:
        models (Dict[str, LinearRegression]): Trained models.
        x_test (pd.DataFrame): Test feature matrix.

    Returns:
        Dict[str, np.ndarray]: Predictions for each target variable.
    """
    return {col: models[col].predict(x_test) for col in models}


def evaluate_model(models: Dict[str, LinearRegression], x_test: pd.DataFrame, y_test: Dict[str, np.ndarray]) -> Dict[
    str, float]:
    """
    Evaluates models using Mean Squared Error.

    Args:
        models (Dict[str, LinearRegression]): Trained models.
        x_test (pd.DataFrame): Test feature matrix.
        y_test (Dict[str, np.ndarray]): True values for the test set.

    Returns:
        Dict[str, float]: Mean Squared Error for each model.
    """
    y_pred = predict(models, x_test)
    return {col: mean_squared_error(y_test[col], y_pred[col]) for col in models}


def pipeline(path_data: str = ""):
    """
    Full data processing and model training pipeline.

    Args:
        path_data (str): Path to the dataset.
    """
    config = load_config("../config.yaml")
    path_data = config["data"]["clean_data_path"]
    path_models = config["models"]["path"]

    logger.info("Creating training set...")
    data_x, data_y = create_training_set(path_data)

    logger.info("Splitting data...")
    x_train, x_test, y_train, y_test = create_splits(data_x, data_y)

    logger.info("Training models...")
    models = fit_model(x_train, y_train, path_models)

    logger.info("Evaluating models...")
    mses = evaluate_model(models, x_test, y_test)

    logger.info(f"Mean Squared Errors: {mses}")


if __name__ == '__main__':
    pipeline()
