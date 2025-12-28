"""Verify that model files load and print environment & model metadata.

Usage:
    python tools/check_models.py

This script is safe to run locally in a trusted environment. It will attempt to load
`xgboost_model.pkl` and `lightgbm_model.pkl` using joblib and print basic info.
"""
import os
import joblib
import importlib

MODELS = ['xgboost_model.pkl', 'lightgbm_model.pkl']
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print('Environment versions:')
for mod in ['python', 'numpy', 'xgboost', 'lightgbm', 'sklearn', 'joblib', 'flask']:
    try:
        if mod == 'python':
            import sys
            print(f"  python: {sys.version.split()[0]}")
        else:
            m = importlib.import_module(mod)
            print(f"  {mod}: {getattr(m, '__version__', 'unknown')}")
    except Exception as e:
        print(f"  {mod}: not installed ({e})")

print('\nModel checks:')
for fname in MODELS:
    path = os.path.join(ROOT, fname)
    if not os.path.exists(path):
        print(f"  {fname}: NOT FOUND at {path}")
        continue
    try:
        m = joblib.load(path)
        cls = m.__class__
        print(f"  {fname}: loaded -> class={cls} module={getattr(m, '__module__', None)}")
        print(f"    supports predict: {hasattr(m, 'predict')}, supports predict_proba: {hasattr(m, 'predict_proba')}")
    except Exception as e:
        print(f"  {fname}: failed to load ({e})")
