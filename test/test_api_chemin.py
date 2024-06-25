import os
import requests
import pandas as pd
from joblib import load

# Définir le chemin de base
base_path = os.path.dirname(os.path.abspath(__file__))

# Charger le preprocessor
path_preprocessor = os.path.abspath(os.path.join(base_path, '../models/preprocessor.joblib'))
print(f"Chemin du preprocessor: {path_preprocessor}")
if not os.path.exists(path_preprocessor):
    raise FileNotFoundError(f"Preprocessor not found at {path_preprocessor}")
preprocessor = load(path_preprocessor)

# Charger les noms des caractéristiques
feature_names_path = os.path.abspath(os.path.join(base_path, "../models/feature_names.txt"))
print(f"Chemin des noms des caractéristiques: {feature_names_path}")
if not os.path.exists(feature_names_path):
    raise FileNotFoundError(f"Feature names file not found at {feature_names_path}")
with open(feature_names_path, "r") as f:
    feature_names = f.read().splitlines()

def test_api_chemin():
    # Charger les données de test
    data_test_path = os.path.abspath(os.path.join(base_path, "../data/processed/processed_data_test_sample.csv"))
    print(f"Chemin des données de test: {data_test_path}")
    if not os.path.exists(data_test_path):
        raise FileNotFoundError(f"Test data file not found at {data_test_path}")
    data_test = pd.read_csv(data_test_path)
    data_test = data_test.loc[:, ~data_test.columns.str.contains('^Unnamed')]
    data_test = data_test[feature_names]
    data_test_preprocessed = preprocessor.transform(data_test)
    data_dict = pd.DataFrame(data_test_preprocessed, columns=feature_names).to_dict(orient='records')
    
    # URL de l'API
    url = "http://127.0.0.1:5000/predict"
    
    # Envoyer la requête POST
    response = requests.post(url, json=data_dict)
    response_json = response.json()
    
    # Vérifier que les probabilités s'additionnent à 1
    probabilities = response_json['probabilities']
    for i, prob in enumerate(probabilities):
        assert abs(sum(prob) - 1.0) < 1e-6, f"Les probabilités ne s'additionnent pas à 1 pour l'instance {i}"

if __name__ == "__main__":
    test_api_chemin()
