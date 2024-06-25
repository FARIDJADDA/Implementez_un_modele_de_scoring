import pytest
import pandas as pd
import joblib
import numpy as np

@pytest.fixture(scope='module')
def data_train():
    """Obtenir les données d'entraînement traitées pour les tests"""
    return pd.read_csv("./data/processed/processed_data_train.csv")

@pytest.fixture(scope='module')
def data_test():
    """Obtenir les données de test traitées pour les tests"""
    return pd.read_csv("./data/processed/processed_data_test_sample.csv")

@pytest.fixture(scope='module')
def feature_names():
    """Obtenir les noms des caractéristiques"""
    with open("./models/feature_names.txt", "r") as f:
        return f.read().splitlines()

@pytest.fixture(scope='module')
def model():
    """Charger le modèle entraîné"""
    return joblib.load("./models/best_model_v2.joblib")

def test_model_predictions(data_test, model, feature_names):
    """Tester que le modèle peut prédire sans erreurs sur les données de test"""
    X_test = data_test[feature_names]
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test), "Le nombre de prédictions ne correspond pas au nombre de données de test"

def test_model_probability_sum(data_test, model, feature_names):
    """Vérifier que la somme des probabilités pour chaque prédiction est égale à 1"""
    X_test = data_test[feature_names]
    probabilities = model.predict_proba(X_test)
    assert np.allclose(np.sum(probabilities, axis=1), 1), "La somme des probabilités pour certaines prédictions n'est pas égale à 1"
