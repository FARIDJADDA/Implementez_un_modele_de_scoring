import sys
import os
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd

# Ajouter le chemin src au sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict_model import load_model, predict, load_pipeline, load_feature_names

app = Flask(__name__)

pipeline = load_pipeline()
model = load_model()
feature_names = load_feature_names()

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json(force=True)
    
    # Debug log pour afficher les données reçues
    print("Données reçues:", data)

    if not data:
        return jsonify({'error': 'Aucune donnée reçue'}), 400

    data = pd.DataFrame(data)

    # Vérifier que les colonnes sont dans le bon ordre et correspondent
    try:
        data = data[feature_names]
    except KeyError as e:
        return jsonify({'error': f'Colonnes manquantes: {str(e)}'}), 400

    # Nettoyer les données pour supprimer les valeurs infinies et NaN
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # Transformer les données
    data_transformed = pipeline.transform(data)

    predictions, probabilities = predict(model, data_transformed)
    
    return jsonify({
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
