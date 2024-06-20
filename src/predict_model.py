import pandas as pd
from data_preparation import load_data
from sklearn.pipeline import Pipeline
import mlflow.sklearn

import os

def load_model(model_path):
    # This function will load the model from the given path
    return mlflow.sklearn.load_model(model_path)

def predict(model, X):
    # Predict function using the loaded model
    return model.predict(X)

if __name__ == "__main__":
    # Load test data
    data_test = load_data('../data/processed/processed_data_test.csv')
    
    # Load the trained model
    model = load_model('../models/best_model_v1') 
    
    # Make predictions
    predictions = predict(model, data_test)
    
    # Ensure the output directory exists
    output_dir = '../output'
    
    # Save predictions to CSV in the output directory
    predictions_file = os.path.join(output_dir, 'predictions.csv')
    pd.DataFrame(predictions).to_csv(predictions_file, index=False)
