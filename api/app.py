# Script principal de l'API Flask.
from flask import Flask, request, jsonify
import pandas as pd
import mlflow.sklearn

app = Flask(__name__)

# Load the trained model
model_path = "../models/best_model_v1"
model = mlflow.sklearn.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame(data['data'])
        predictions = model.predict(df)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
