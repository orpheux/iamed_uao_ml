import os, sys
from pycaret.classification import load_model

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

model = load_model(resource_path("modelo/best_iris_classifier"))

