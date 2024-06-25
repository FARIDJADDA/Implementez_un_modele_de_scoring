import os
import requests
import pandas as pd
from joblib import load

def test_api_with_new_sample():
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

    # Sélectionner un sous-échantillon de données
    sample_data = data.sample(n=10, random_state=2)  # Sélectionner 10 lignes aléatoires
    
    # Réorganiser les colonnes pour qu'elles correspondent à celles utilisées lors de l'entraînement
    sample_data = sample_data[feature_names]

    # Appliquer le pipeline de prétraitement
    sample_data_preprocessed = preprocessor.transform(sample_data)

    # Convertir les données transformées en dictionnaire avant l'envoi
    sample_data_dict = pd.DataFrame(sample_data_preprocessed, columns=feature_names).to_dict(orient='records')

    # URL de l'API
    url = "http://127.0.0.1:5000/predict"

    # Envoyer la requête POST
    response = requests.post(url, json=sample_data_dict)

    # Afficher la réponse
    print("Response for new sample:", response.json())

if __name__ == "__main__":
    test_api_with_new_sample()
