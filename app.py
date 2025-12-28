from flask import Flask, render_template, request, jsonify, abort
import joblib
import numpy as np
import logging
import os
from functools import wraps

# Optional CORS and simple token auth
try:
    from flask_cors import CORS
except Exception:
    CORS = None

app = Flask(__name__, template_folder='Electricity/templates')

# Enable CORS only if requested via env vars
CORS_ENABLED = os.environ.get('CORS_ENABLED', 'False').lower() in ('1','true','yes')
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')
if CORS is not None and CORS_ENABLED:
    origins = CORS_ORIGINS if CORS_ORIGINS == '*' else [o.strip() for o in CORS_ORIGINS.split(',')]
    CORS(app, resources={r"/*": {"origins": origins}})


# Simple token auth settings: evaluated at request-time so tests can toggle env vars safely
def require_token(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        enabled = os.environ.get('API_AUTH_ENABLED', 'False').lower() in ('1','true','yes')
        if not enabled:
            return fn(*args, **kwargs)
        api_token = os.environ.get('API_TOKEN')
        auth = request.headers.get('Authorization','')
        if auth.startswith('Bearer '):
            provided = auth.split(' ',1)[1]
        else:
            provided = auth
        if not api_token or provided != api_token:
            abort(401, description='Unauthorized')
        return fn(*args, **kwargs)
    return wrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper to load models safely
def load_model(path):
    try:
        model = joblib.load(path)
        logger.info(f"Loaded model: {path}")
        return model
    except Exception as e:
        logger.exception(f"Failed to load model {path}: {e}")
        return None

MODEL_DIR = os.environ.get('MODEL_DIR', 'Electricity')

xgb_model = load_model(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
lgbm_model = load_model(os.path.join(MODEL_DIR, 'lightgbm_model.pkl'))

# Feature schema: (name, type, min, max)
FEATURES = [
    ('mean_consumption', float, 0, None),
    ('std_consumption', float, 0, None),
    ('cv', float, 0, None),
    ('sudden_drop_count', int, 0, None),
    ('min_consumption', float, 0, None),
    ('num_below_mean', int, 0, None),
    ('skewness', float, None, None),
    ('iqr', float, 0, None),
    ('max_drop_pct', float, 0, 100)
]

import math

def parse_input(data):
    """Parse and validate incoming data dict. Returns (values_list, errors_list)."""
    values = []
    errors = []

    for name, typ, minv, maxv in FEATURES:
        field_errors = []

        if name not in data:
            field_errors.append(f"Missing field: {name}")
            errors.extend(field_errors)
            continue

        raw = data.get(name)
        if raw is None or raw == '':
            field_errors.append(f"Empty value for {name}")
            errors.extend(field_errors)
            continue

        try:
            if typ == int:
                # Accept numeric strings or float-like values for ints
                val = int(float(raw))
            else:
                val = float(raw)
        except Exception:
            field_errors.append(f"Invalid value for {name}: {raw}")
            errors.extend(field_errors)
            continue

        # Reject NaN or infinite values
        if not math.isfinite(val):
            field_errors.append(f"{name} must be a finite number")

        if minv is not None and val < minv:
            field_errors.append(f"{name} must be >= {minv}")
        if maxv is not None and val > maxv:
            field_errors.append(f"{name} must be <= {maxv}")

        if field_errors:
            errors.extend(field_errors)
        else:
            values.append(val)

    return values, errors

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'ok': True,
        'model_dir': MODEL_DIR,
        'models': {
            'xgboost': {'loaded': bool(xgb_model)},
            'lightgbm': {'loaded': bool(lgbm_model)}
        }
    })

@app.route('/ready')
def ready():
    """Readiness endpoint returns 200 only if models are loaded."""
    if xgb_model is None or lgbm_model is None:
        return jsonify({'ready': False, 'reason': 'models not loaded'}), 503
    return jsonify({'ready': True})

@app.route('/models')
def models():
    def info(m):
        if m is None:
            return {'loaded': False}
        return {'loaded': True, 'supports_proba': hasattr(m, 'predict_proba')}

    return jsonify({'xgboost': info(xgb_model), 'lightgbm': info(lgbm_model)})

@app.route('/predict', methods=['POST'])
@require_token
def predict():
    try:
        # Accept JSON or form data
        incoming = request.get_json(silent=True)
        if incoming is None:
            incoming = request.form.to_dict()

        # Default model_type to 'xgboost' if not provided
        model_type = incoming.get('model_type', 'xgboost')

        # Parse inputs
        features_values, errors = parse_input(incoming)
        if errors:
            return jsonify({'error': 'Input validation failed', 'details': errors}), 400

        # Ensure feature order matches training
        features_array = np.array(features_values).reshape(1, -1)

        # Choose model
        if model_type == 'xgboost':
            model = xgb_model
            model_name = 'XGBoost'
        elif model_type == 'lightgbm':
            model = lgbm_model
            model_name = 'LightGBM'
        else:
            return jsonify({'error': 'Invalid model_type; use xgboost or lightgbm'}), 400

        if model is None:
            return jsonify({'error': f'Model {model_type} not loaded on server'}), 503

        # Run prediction
        pred_raw = model.predict(features_array)[0]
        try:
            pred = int(pred_raw)
        except Exception:
            pred = int(np.round(pred_raw))

        label = 'Theft Detected' if pred == 1 else 'No Theft Detected'

        probabilities = None
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(features_array)[0]
                probabilities = {
                    'no_theft': {'prob': float(probs[0]), 'pct': f"{probs[0]*100:.2f}%"},
                    'theft': {'prob': float(probs[1]), 'pct': f"{probs[1]*100:.2f}%"}
                }
                confidence = probabilities['theft']['pct'] if pred == 1 else probabilities['no_theft']['pct']
            except Exception:
                logger.exception('predict_proba failed')
                probabilities = None

        response = {
            'prediction': label,
            'model_used': model_name,
            'confidence': confidence if confidence is not None else 'N/A',
            'probabilities': probabilities
        }

        return jsonify(response)

    except Exception as e:
        logger.exception('Unexpected error during prediction')
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Respect PORT and DEBUG env vars if provided
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() in ('1', 'true', 'yes')
    app.run(debug=debug, host='0.0.0.0', port=port)
