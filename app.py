"""
Root entry point forwarding to frontend/app.py.
Allows running:
    streamlit run app.py
or
    streamlit run frontend/app.py
"""
import os
import sys

# Ensure current directory is in path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Run the frontend application
from frontend import app  # noqa
