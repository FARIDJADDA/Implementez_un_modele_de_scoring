import pandas as pd
import json
import requests
from joblib import load
import os

# Charger le preprocessor
base_path = os.path.dirname(os.path.abspath(__file__))
preprocessor_path = os.path.abspath(os.path.join(base_path, '../models/preprocessor.joblib'))
preprocessor = load(preprocessor_path)

# Charger les noms des caractéristiques
feature_names_path = os.path.abspath(os.path.join(base_path, "../models/feature_names.txt"))
with open(feature_names_path, "r") as f:
    feature_names = f.read().splitlines()

# Charger les données de test
data_test_path = os.path.abspath(os.path.join(base_path, "../data/processed/processed_data_test_sample.csv"))
data_test = pd.read_csv(data_test_path)

# Exclure les colonnes inutiles
data_test = data_test.loc[:, ~data_test.columns.str.contains('^Unnamed')]

# Sélectionner un échantillon de données (par exemple, la première ligne)
sample_data = data_test.sample(1)

# Prétraiter les données
sample_data_preprocessed = preprocessor.transform(sample_data[feature_names])

# Convertir le DataFrame en JSON
sample_data_json = pd.DataFrame(sample_data_preprocessed, columns=feature_names).to_json(orient='records')

# Afficher le JSON
print("Sample Data JSON:", sample_data_json)

# URL de l'API déployée sur Azure
url = "https://my-ml-api-app.azurewebsites.net/predict"

# Envoyer la requête POST à l'API
response = requests.post(url, headers={"Content-Type": "application/json"}, data=sample_data_json)

# Afficher la réponse de l'API
print("Response status code:", response.status_code)
try:
    print("Response JSON:", response.json())
except json.decoder.JSONDecodeError as e:
    print("JSON Decode Error:", str(e))
    print("Response Content:", response.text)
