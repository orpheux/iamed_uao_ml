from model import model
from pycaret.classification import predict_model
import pandas as pd

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    """Recibe 4 valores de entrada y devuelve la clase predicha del iris."""
    input_data = pd.DataFrame([{
        'sepal length (cm)': float(sepal_length),
        'sepal width (cm)': float(sepal_width),
        'petal length (cm)': float(petal_length),
        'petal width (cm)': float(petal_width)
    }])
    prediction = predict_model(model, data=input_data)
    return prediction['prediction_label'].iloc[0]
