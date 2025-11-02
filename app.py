from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os
import joblib

# Import XGBoost before loading the model (required for joblib deserialization)
try:
    import xgboost as xgb
    print("XGBoost imported successfully")
except ImportError:
    print("Warning: XGBoost not imported. Model loading may fail.")

app = Flask(__name__, static_folder='electricity_fraud')
CORS(app)

# Load the XGBoost model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'electricity_fraud', 'xgb_model_compressed.pkl')

try:
    # Ensure XGBoost is imported before loading (critical for joblib deserialization)
    import xgboost as xgb
    
    # Verify model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    print(f"Loading model from: {MODEL_PATH}")
    
    # Load the model (joblib requires XGBoost to be imported first)
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully: {type(model).__name__}")
    
    # Verify it's an XGBoost model
    if hasattr(model, 'get_booster'):
        num_features = model.get_booster().num_features()
        print(f"XGBoost model verified - Feature count: {num_features}")
    elif hasattr(model, 'n_features_in_'):
        print(f"XGBoost model verified - Feature count: {model.n_features_in_}")
    else:
        print(f"Warning: Model type {type(model).__name__} may not be XGBoost")
        
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please ensure xgb_model_compressed.pkl exists in the electricity_fraud/ directory")
    model = None
except ImportError as e:
    print(f"Error: XGBoost import failed - {e}")
    print("Please ensure xgboost is installed: pip install xgboost")
    model = None
except Exception as e:
    print(f"Error loading model: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    model = None

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('electricity_fraud', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict fraud probability using the loaded model"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            }), 500
        
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'success': False
            }), 400
        
        # Required features in the exact order expected by the model
        required_features = [
            'mean_consumption',
            'cv',
            'num_zeros',
            'trend_slope',
            'sudden_drop_count',
            'num_below_mean',
            'skewness',
            'iqr',
            'max_drop_pct'
        ]
        
        # Validate all required features are present
        missing_features = [f for f in required_features if f not in data or data[f] is None]
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {", ".join(missing_features)}',
                'success': False,
                'missing_features': missing_features
            }), 400
        
        # Extract features in the exact order expected by the model
        # Convert to appropriate types (int for counts, float for others)
        features = [
            float(data.get('mean_consumption', 0)),
            float(data.get('cv', 0)),
            int(data.get('num_zeros', 0)),
            float(data.get('trend_slope', 0)),
            int(data.get('sudden_drop_count', 0)),
            int(data.get('num_below_mean', 0)),
            float(data.get('skewness', 0)),
            float(data.get('iqr', 0)),
            float(data.get('max_drop_pct', 0))
        ]
        
        # Validate feature count matches model expectation
        # XGBoost models may use n_features_in_ or get_booster().num_features()
        expected_features = None
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
        elif hasattr(model, 'get_booster'):
            try:
                expected_features = model.get_booster().num_features()
            except Exception:
                pass
        
        if expected_features and expected_features != len(features):
            return jsonify({
                'error': f'Feature count mismatch: model expects {expected_features} features, got {len(features)}',
                'success': False
            }), 400
        
        # Convert to numpy array and reshape for prediction
        features_array = np.array(features, dtype=np.float64).reshape(1, -1)
        
        # Make binary prediction (0 = Normal, 1 = Fraud)
        prediction = float(model.predict(features_array)[0])

        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_array)[0]
            # For binary classification, proba[1] is the probability of fraud (flag=1)
            fraud_probability = float(proba[1] if len(proba) > 1 else proba[0])
        else:
            # If no predict_proba, use prediction directly (0 or 1)
            fraud_probability = float(prediction)
        
        # Calculate fraud score (0-100)
        fraud_score = fraud_probability * 100 if fraud_probability <= 1 else fraud_probability
        
        # Binary classification: 0 = Normal, 1 = Fraud
        # Use 0.5 as threshold for binary decision
        is_fraud = int(prediction) == 1
        classification = 'fraud' if is_fraud else 'normal'
        
        # Determine risk level based on binary prediction and probability
        if is_fraud:
            risk_level = 'FRAUD DETECTED'
        else:
            risk_level = 'NORMAL'
        
        # Generate indicators based on features
        indicators = []
        # Note: flag is the prediction result, not an input feature
        if data.get('num_zeros', 0) > 10:
            indicators.append('Excessive zero readings detected')
        if data.get('cv', 0) > 1.5:
            indicators.append('High consumption variability')
        if data.get('sudden_drop_count', 0) > 5:
            indicators.append('Multiple sudden consumption drops')
        if data.get('max_drop_pct', 0) > 70:
            indicators.append('Severe consumption drop detected')
        if data.get('trend_slope', 0) < -0.5:
            indicators.append('Strong declining consumption trend')
        if abs(data.get('skewness', 0)) > 2:
            indicators.append('Abnormal consumption distribution')
        
        # Calculate confidence based on prediction probability
        confidence = min(85 + (fraud_score * 0.15), 99)
        
        return jsonify({
            'success': True,
            'prediction': {
                'score': round(fraud_score, 2),
                'riskLevel': risk_level,
                'classification': classification,
                'indicators': indicators,
                'confidence': round(confidence, 1),
                'fraud_probability': round(fraud_probability, 4) if fraud_probability <= 1 else round(fraud_probability / 100, 4)
            },
            'customer_id': data.get('customer_id', 'N/A')
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)

