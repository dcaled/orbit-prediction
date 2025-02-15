# README

## Installation
To set up the environment, install the required dependencies using Anaconda. Run the following command:

```sh
conda install --file requirements.txt
```

## 1. Data Processing Module
The data processing module contains a configuration file where the user should specify the data path. By default, the data path is set to:

```
../data/{space-object-id}/raw
```

SP3-c raw files should be placed in this directory.

### Running the Data Processing
To run the data processing, navigate to the `data_processing` folder and execute the following command:

```sh
cd data_processing
python preprocess.py
```

## 3. API Deployment

The Orbit prediction service is implemented using FastAPI. The API consists of a single endpoint that allows inference 
requests for the satellite position in a given datetime.

### 3.1. Request

The endpoint expects as input a JSON object containing the space-object identifier and the number of predictions:

Example:
```
{
  "space_object_id": "larets",
  "number_of_predictions": 3
}
```

The API response will be the predicted positions of the satellite in the following example format:
```
{
  "positions": [
    {
      "epoch": "2025-02-15T11:33:11.036209",
      "x": 1.2469855511354737,
      "y": 2.858835514866617,
      "z": 2.4171653647810856
    },
    {
      "epoch": "2025-02-15T11:33:11.036209",
      "x": 1.1569145397294214,
      "y": 1.6031156355035945,
      "z": 2.097456147037125
    },
    {
      "epoch": "2025-02-15T11:33:11.036209",
      "x": 1.756933762210285,
      "y": 1.2102738414184224,
      "z": 1.8378660689846575
    }
  ]
}
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