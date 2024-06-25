import os
import requests
import pandas as pd
from joblib import load
import numpy as np

def test_api_with_extreme_cases():
    # Obtenir le chemin absolu du répertoire actuel
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construire le chemin absolu des fichiers nécessaires
    data_path = os.path.join(current_dir, "../data/processed/processed_data_test.csv")
    feature_names_path = os.path.join(current_dir, "../models/feature_names.txt")
    preprocessor_path = os.path.join(current_dir, "../models/preprocessor.joblib")

    # Charger le dataset complet
    data = pd.read_csv(data_path)

    # Exclure les colonnes inutiles
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Charger les noms des caractéristiques
    with open(feature_names_path, "r") as f:
        feature_names = f.read().splitlines()

    # Charger le pipeline de prétraitement
    preprocessor = load(preprocessor_path)

    # URL de l'API
    url = "http://127.0.0.1:5000/predict"

    # Ajouter des valeurs extrêmes
    extreme_data = data.copy()
    extreme_data.iloc[0] = np.inf
    extreme_data.iloc[1] = -np.inf
    extreme_data.iloc[2] = np.nan

    # Appliquer le pipeline de prétraitement
    extreme_data_preprocessed = preprocessor.transform(extreme_data.fillna(0).replace([np.inf, -np.inf], 0))

    # Convertir les données transformées en dictionnaire avant l'envoi
    extreme_data_dict = pd.DataFrame(extreme_data_preprocessed, columns=feature_names).to_dict(orient='records')

    # Envoyer la requête POST
    response = requests.post(url, json=extreme_data_dict)

    # Afficher la réponse
    print("Response for extreme cases:", response.json())

if __name__ == "__main__":
    test_api_with_extreme_cases()
