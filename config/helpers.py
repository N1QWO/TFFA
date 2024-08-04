import sys
import os

def add_base_path():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if base_path not in sys.path:
        sys.path.append(base_path)