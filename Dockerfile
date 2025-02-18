FROM python:3.12-slim

# Set the working directory
WORKDIR /model_api

# Copy the required files
COPY requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the FastAPI default port
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "model_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
