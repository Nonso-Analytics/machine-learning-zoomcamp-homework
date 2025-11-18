<<<<<<< HEAD
# Water Potability Prediction 

A machine learning pipeline for predicting water potability (safety) based on quality parameters. This project demonstrates the full ML workflow from data exploration to cloud deployment.

## Project Overview

### Problem Statement
Access to safe drinking water is critical for public health. Traditional lab testing is expensive and time-consuming (24-48 hours). This ML solution provides instant water safety predictions based on 9 key quality parameters.

### Solution
Binary classification model that predicts whether water is potable (safe) or not potable (unsafe) with confidence scores.

**Impact**: Enable rapid water quality assessment in areas without lab facilities and emergency response situations.

## Dataset

- **Source**: [Kaggle - Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- **Size**: 3,276 water samples
- **Features**: 9 water quality parameters
- **Target**: Binary (Potable: 1, Not Potable: 0)

### Features
1. **pH** - pH level (0-14)
2. **Hardness** - Water hardness (mg/L)
3. **Solids** - Total dissolved solids (ppm)
4. **Chloramines** - Chloramine concentration (ppm)
5. **Sulfate** - Sulfate concentration (mg/L)
6. **Conductivity** - Electrical conductivity (μS/cm)
7. **Organic_carbon** - Organic carbon content (ppm)
8. **Trihalomethanes** - THM concentration (μg/L)
9. **Turbidity** - Water clarity (NTU)

## Quick Start

### Prerequisites
```bash
Python 3.8+
pip
Docker (for containerization)
Fly.io CLI (for cloud deployment)
```

### 1. Setup Environment

```bash
# Clone or create project directory
mkdir midterm-project && cd midterm-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

1. Go to https://www.kaggle.com/datasets/adityakadiwal/water-potability
2. Download `water_potability.csv`

### 3. Train Models

```bash
# Run complete training pipeline
python train.py
```

This will:
- Load and preprocess data
- Train 6 different models (baseline)
- Perform hyperparameter tuning
- Save the best model
- Generate comparison metrics

### 4. Make Predictions

```bash
# Interactive mode
python predict.py --mode interactive

# Example predictions
python predict.py --mode example

# Batch predictions from CSV
python predict.py --mode csv --csv data.csv --output predictions.csv
```

### 5. Run API Locally

```bash
# Start API server
python serve.py

# Or using uvicorn directly
uvicorn serve:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:
- **Main**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### Key Insights from EDA
- Class imbalance: 61% not potable, 39% potable
- Missing values: pH (15%), Sulfate (24%), Trihalomethanes (12%)
- No strong multicollinearity (max correlation < 0.5)
- Tree-based models outperform linear models

## Docker Deployment

### Build Docker Image

```bash
# Build image
docker build -t water-potability-api .

# Run container
docker run -it --rm -p 8000:8000 water-potability-api

# Test
curl http://localhost:8000/health
```


## Cloud Deployment (Fly.io)

### Setup Fly.io

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

fly auth signup
fly launch --generate-name
fly deploy
```



## API Usage Examples

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ph": 7.0,
    "Hardness": 200.0,
    "Solids": 20000.0,
    "Chloramines": 7.5,
    "Sulfate": 350.0,
    "Conductivity": 400.0,
    "Organic_carbon": 15.0,
    "Trihalomethanes": 70.0,
    "Turbidity": 4.0
  }'
```

### Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "ph": 7.0,
    "Hardness": 200.0,
    "Solids": 20000.0,
    "Chloramines": 7.5,
    "Sulfate": 350.0,
    "Conductivity": 400.0,
    "Organic_carbon": 15.0,
    "Trihalomethanes": 70.0,
    "Turbidity": 4.0
}

response = requests.post(url, json=data)
print(response.json())
```



## Testing

```bash
# Test API endpoints
python test.py

```


=======
# machine-learning-zoomcamp-homework
This repository contains all homework from my participation in Machine Learning Zoomcamp 2025
>>>>>>> c62a52cbb7d32968108b3fe4e2624f511c9699d3
