# README

## Installation
To set up the environment, install the required dependencies using Anaconda. Run the following command:

```sh
conda install --file requirements.txt
```

## 1. Data Processing Module
The data processing module contains a configuration file where the user should specify the data path. By default, the 
data path is set to:

```
data:
   path_raw_data: data/{space-object-id}/raw
```

SP3-c raw files should be placed in this directory.

You should also specify the path in which the clean data will be stored. Example:

```
data:
   path_clean_data: data/{space-object-id}/clean
```

### 1.1. Running the Data Processing
To run the data processing, navigate to the root folder and execute the following command:

```sh
python data_processing\preprocess.py
```

## 2. Model training

This project utilizes XGBoost Regression optimized with Optuna to predict the future positions of a space object based on historical trajectory data. Below is the configuration used for training and storing models.

To train a model, you should configure the file `config.yaml` with the following parameters:

- `path`: Specifies where the trained models should be saved. `{space-object-id}` is a placeholder for the unique identifier of the space-object. 
- `path_first_position`: The location of the file that stores the first recorded position of the space-object.
- `n_trials`: Defines the number of trials Optuna will run to find the best hyperparameters.
- `hyperparameters`: Defines the range of values that Optuna will explore when tuning the hyperparameters of the XGBoost model.
  - `n_estimators` (100 - 1000): Number of decision trees (boosted trees) in the model.
  - `learning_rate` (0.01 - 0.3): Controls how much each tree contributes to the final prediction.
  - `max_depth` (3 - 10): Maximum depth of each decision tree.
  - `subsample` (0.5 - 1.0): The fraction of training samples used to build each tree.
  - `colsample_bytree` (0.5 - 1.0): The fraction of features (columns) used per tree.

Example:
```
models:
  path: data/{space-object-id}/models
  path_first_position: data/{space-object-id}/first_position.json
  n_trials: 25
  hyperparameters:
    n_estimators:
      min: 100
      max: 1000
    learning_rate:
      min: 0.01
      max: 0.3
    max_depth:
      min: 3
      max: 10
    subsample:
      min: 0.5
      max: 1.0
    colsample_bytree:
      min: 0.5
      max: 1.0
```


### 2.1. Running model training

To run the data processing, navigate to the root folder and execute the following command:

```sh
python model_training\train.py
```

Three different models will be created, one for predicting each space-object's orbit position (`x,y,z`).

## 3. API Deployment

The Orbit prediction service is implemented using FastAPI. The API consists of a single endpoint that allows inference 
requests for the satellite position in a given datetime.

### 3.1. Request

The endpoint expects as input a JSON object containing the space-object identifier and the last position of this object:

Example:
```
{
  "space_object_id": "larets",
  "epoch": "2025-02-18T07:00:34.583Z",
  "pos_x": -2.787127,
  "pos_y": 5994.151251,
  "pos_z": 3726.167103,
  "vel_x": 17226.162,
  "vel_y": -39257.044,
  "vel_z": 62793.22
}
```

The API response will be the predicted positions of the satellite in the following example format:
```
{
  "space_object_id": "larets",
  "positions": [
    {
      "epoch": "2025-02-18T07:00:34.583000",
      "pos_x": 295.064743646789,
      "pos_y": 5178.821646016164,
      "pos_z": 4780.961592038952
    },
      ...
    {
      "epoch": "2025-02-18T07:09:34.583000",
      "pos_x": 2748.6403603933686,
      "pos_y": -1521.157660367164,
      "pos_z": 13447.5371847854
    }
  ]
}
```

By default, the number of future position predictions and the time step between predictions are respectively defined as
`10` and `60` seconds. These parameters can be modified in the configuration file:

```
inference:
  number_of_predictions: 10
  delta: 60
```

### 3.2. Running the API (locally)

To run your FastAPI application, follow these steps:
1. Enter the project root folder
2. Run the FastAPI Application using uvicorn to start the FastAPI server. 
```
uvicorn model_api.app:app --host 0.0.0.0 --port 8000
```
3. Once running, access the API documentation at:
   - Swagger UI: http://127.0.0.1:8000/docs
   - Redoc: http://127.0.0.1:8000/redoc
   
### 3.3. Running the API (via Docker)

1. Enter the project root folder
2. Build the container by running
```
docker build . -t model_api
```

3. Run the image:
```
docker run -p 8000:8000 -d model_api
 ```