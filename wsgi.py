import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when Gunicorn runs from different CWD
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide helpful debug logging early to aid platform troubleshooting
import logging
logger = logging.getLogger('wsgi_startup')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.debug(f"Current working dir: {os.getcwd()}")
logger.debug(f"sys.path[0:5]: {sys.path[0:5]}")
logger.debug(f"Project root (inserted): {ROOT}")
logger.debug(f"Listing project root files: {list(ROOT.iterdir())}")

# Import the Flask app object from app.py
try:
    from app import app
except Exception as e:
    # Re-raise with more context for easier debugging in logs
    logger.exception("Failed to import Flask app from 'app.py'.")
    raise ImportError(f"Failed to import Flask app from 'app.py': {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
