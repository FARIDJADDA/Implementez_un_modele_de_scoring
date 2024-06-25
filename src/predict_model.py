import os
import joblib

def load_pipeline():
    base_path = os.path.dirname(os.path.abspath(__file__))
    pipeline_path = os.path.join(base_path, '../models/preprocessor.joblib')
    return joblib.load(pipeline_path)

def load_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, '../models/best_model_v2.joblib')
    return joblib.load(model_path)

def load_feature_names():
    base_path = os.path.dirname(os.path.abspath(__file__))
    feature_names_path = os.path.join(base_path, "../models/feature_names.txt")
    with open(feature_names_path, "r") as f:
        feature_names = f.read().splitlines()
    return feature_names

def predict(model, data):
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)
    return predictions, probabilities
