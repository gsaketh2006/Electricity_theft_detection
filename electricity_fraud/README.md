# Electricity Fraud Detection System

A Flask-based web application for detecting electricity fraud using an XGBoost machine learning model.

## Features

- **Web Interface**: User-friendly HTML interface for entering consumption data
- **Machine Learning Backend**: Uses XGBoost model for fraud detection
- **Real-time Analysis**: Instant fraud probability assessment
- **Risk Classification**: Categorizes cases as High Risk, Medium Risk, or Low Risk

## Installation

1. Install Python dependencies:
```bash
pip install -r electricity_fraud/requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## API Endpoints

### POST /predict
Predict fraud probability based on consumption data.

**Request Body:**
```json
{
  "customer_id": "MTR-2024-1234",
  "mean_consumption": 150.5,
  "cv": 0.3,
  "num_zeros": 2,
  "trend_slope": -0.1,
  "sudden_drop_count": 1,
  "num_below_mean": 10,
  "skewness": 0.5,
  "iqr": 60.0,
  "max_drop_pct": 20.0
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "score": 15.5,
    "riskLevel": "LOW RISK",
    "classification": "legitimate",
    "indicators": [],
    "confidence": 87.3,
    "fraud_probability": 0.155
  },
  "customer_id": "MTR-2024-1234"
}
```

### GET /health
Check the health status of the API and model.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Model Features

The model expects the following 9 features to predict the fraud flag:
1. `mean_consumption` - Mean consumption in kWh
2. `cv` - Coefficient of variation
3. `num_zeros` - Number of zero readings
4. `trend_slope` - Consumption trend slope
5. `sudden_drop_count` - Number of sudden drop incidents
6. `num_below_mean` - Number of readings below mean
7. `skewness` - Skewness of distribution
8. `iqr` - Interquartile range
9. `max_drop_pct` - Maximum drop percentage

**Note:** The `flag` is the target variable (predicted output), not an input feature.

## Deployment on Render

This application is configured for deployment on Render.

### Steps to Deploy:

1. **Push your code to GitHub/GitLab/Bitbucket**

2. **Connect to Render:**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your repository

3. **Configure the service:**
   - **Name**: `electricity-fraud-detection` (or any name you prefer)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r electricity_fraud/requirements.txt`
   - **Start Command**: `gunicorn app:app`

4. **Environment Variables** (optional):
   - `FLASK_ENV`: `production` (for production mode)
   - `PYTHON_VERSION`: `3.11.0` (if needed)

5. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically detect the `render.yaml` file and configure accordingly
   - Wait for the build to complete

6. **Access your app:**
   - Your app will be available at: `https://your-app-name.onrender.com`

### Manual Deployment (without render.yaml):

If you prefer manual configuration, use these settings:
- **Build Command**: `pip install -r electricity_fraud/requirements.txt`
- **Start Command**: `gunicorn app:app`
- **Environment**: Python 3

### Important Notes:

- The app automatically uses the `PORT` environment variable provided by Render
- The `xgb_model_compressed.pkl` file must be included in your repository
- Make sure all files are committed before deploying
- Render provides free tier with automatic sleep after inactivity

## Local Development

For local development, you can run:

```bash
python app.py
```

For production-like local testing:

```bash
gunicorn app:app
```

## Requirements

- Python 3.7+
- Flask 3.0.0
- Flask-CORS 4.0.0
- NumPy 1.26.2
- scikit-learn 1.3.2
- XGBoost 2.0.3
- joblib 1.3.2
- gunicorn 21.2.0 (for production deployment)

