# Electricity Theft Detection using Machine Learning

A machine learning–based system designed to identify electricity theft by analyzing consumer electricity usage patterns. This project leverages advanced machine learning and generative modeling techniques to accurately detect abnormal behavior and support power distribution companies in reducing losses.

---

## Project Overview

Electricity theft causes major financial losses and operational inefficiencies for utility providers. Traditional rule-based detection systems are limited, inaccurate, and difficult to scale.

This project introduces a **data-driven approach** that uses supervised machine learning models combined with **WGAN-GP (Wasserstein GAN with Gradient Penalty)** to handle class imbalance and improve prediction accuracy.

---

## Objectives

- Detect electricity theft using machine learning techniques  
- Handle highly imbalanced datasets using generative models  
- Compare multiple ML classifiers  
- Deploy a real-time prediction system  

---

## Project Structure

```
Electricity_Theft_Detection/
│
├── Electricity/
│ ├── templates/
│ │ |__ index.html
│ │
│ ├── lightgbm_model.pkl
│ └── xgboost_model.pkl
│
├── tools/
│ ├── check_models.py
│ └── poll_ready.py
│
├── Work/
│ ├── EDA.ipynb
│ ├── Model_Evaluation.ipynb
│ └── preprocessing.ipynb
│
├── app.py
├── wsgi.py
├── start.sh
├── Procfile
├── render.yaml
├── requirements.txt
├── runtime.txt
├── .env.example
└── README.md
```

---

## How to Run the Project

### Clone the Repository
```bash
git clone https://github.com/your-username/Electricity_Theft_Detection.git
cd Electricity_Theft_Detection

python -m venv venv

For Windows: venv\Scripts\activate

pip install -r requirements.txt

python app.py

http://127.0.0.1:5000
```
## Dataset Information

**Dataset Type:** Electricity Consumption Dataset  
**Target Variable:** Theft (0 – Normal, 1 – Theft)
**Dataset Link:** https://drive.google.com/file/d/1mlf4Sn0J9-0EPd0tYC6Rx7ZPGn-Jw0U0/view?usp=drive_link

### Key Features
- Mean electricity consumption  
- Standard deviation  
- Sudden drop in consumption  
- Variance and skewness  
- Consumption pattern statistics  

---

## Methodology

## Data Preprocessing & Feature Engineering

The original dataset contained **customer-wise electricity consumption records with timestamps**, including:

- Customer ID  
- Date / Time of consumption  
- Raw energy usage values  

Since machine learning models cannot directly learn effectively from raw time-series data in this format, the data was transformed into **statistical and behavioral features** that capture consumption patterns over time.

### Feature Transformation Approach

For each customer, historical consumption data was aggregated and converted into the following meaningful features:

#### Statistical Features
- **mean_consumption** – Average electricity usage  
- **median_consumption** – Median consumption value  
- **max_consumption** – Maximum recorded consumption  
- **min_consumption** – Minimum recorded consumption  
- **std_consumption** – Standard deviation of consumption  
- **cv** – Coefficient of variation (std / mean)  
- **iqr** – Interquartile range of consumption  
- **skewness** – Skewness of the consumption distribution  

#### Behavioral & Anomaly-Based Features
- **sudden_drop_count** – Number of sudden drops in consumption  
- **max_drop_pct** – Maximum percentage drop observed  
- **num_below_mean** – Count of readings below the mean consumption  
- **num_zeros** – Number of zero-consumption readings  
- **trend_slope** – Consumption trend over time (increasing / decreasing)  

### Why This Transformation Was Necessary

- Removes dependency on raw timestamps  
- Captures long-term usage behavior per customer  
- Highlights abnormal and suspicious consumption patterns  
- Improves model performance and interpretability  
- Makes the data suitable for tree-based and boosting models  

This engineered feature set enables the machine learning models to effectively distinguish between **normal consumption behavior** and **potential electricity theft patterns**.

---


### 3. Model Training
- Train–test split (80% / 20%)
- Stratified sampling
- Two supervised learning models trained:
  - **XGBoost**
  - **LightGBM**

---

## Model Performance

### XGBoost Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| 0.0  | 0.92 | 1.00 | 0.96 | 7751 |
| 1.0  | 1.00 | 0.91 | 0.95 | 7751 |
| **Accuracy** |  |  | **0.95** | 15502 |
| **ROC-AUC** |  |  | **0.9772** |  |

---

### LightGBM Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| 0.0  | 0.92 | 1.00 | 0.96 | 7751 |
| 1.0  | 1.00 | 0.91 | 0.95 | 7751 |
| **Accuracy** |  |  | **0.95** | 15502 |
| **ROC-AUC** |  |  | **0.9774** |  |

---

## Model Comparison Summary

| Model     | Accuracy | ROC-AUC |
|-----------|----------|---------|
| XGBoost   | 95.45%   | 0.9772  |
| LightGBM  | 95.41%   | 0.9774  |

**Selected Model:** Both models LightGBM and XGBoost
**Reason:** Because both having similar accuracy and both are faster

---

## Why Use Machine Learning for Electricity Theft Detection?

- Identifies complex and hidden usage patterns  
- Reduces dependency on manual inspections  
- Scales efficiently for large datasets  
- Adapts to changing consumer behavior  
- Improves detection accuracy and reliability  

---

## Deployment

- Model serialized using `pickle`
- Backend built with **Flask**
- Web interface for real-time predictions
- Deployed using **Render**

---

## Tech Stack

| Category | Tools |
|--------|------|
| Programming Language | Python |
| Machine Learning | XGBoost, LightGBM |
| Imbalance Handling | WGAN-GP |
| Libraries | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Backend | Flask |
| Deployment | Render |

---

## Future Enhancements

- Real-time smart meter data integration  
- Explainable AI using SHAP / LIME  
- Automated retraining pipeline  
- Mobile and dashboard-based monitoring  
- Advanced anomaly detection models  

---

## Authors & Contributors

**Project Title:** Electricity Theft Detection using Machine Learning  

**Contributors:**  
- Guggilam Leela Naga Sai Sri Saketh  
- Seshagiri Bharadwaj Sai  
- Kanaparti Dhanush
- Hrushikesh Bhaskar Gopale  

---

*If you find this project useful, consider giving it a ⭐ on GitHub!*
