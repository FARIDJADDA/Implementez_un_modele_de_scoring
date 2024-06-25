import pytest
import pandas as pd

@pytest.fixture(scope='module')
def data_train():
    """
    Obtenir les données d'entraînement traitées pour les tests
    """
    return pd.read_csv("./data/processed/processed_data_train.csv")

@pytest.fixture(scope='module')
def data_test():
    """Obtenir les données de test traitées pour les tests"""
    return pd.read_csv("./data/processed/processed_data_test_sample.csv")

def test_train_duplicates(data_train):
    """Tester si les doublons dans les données d'entraînement sont absents"""
    duplicates = data_train[data_train.duplicated()]
    assert duplicates.empty

def test_test_duplicates(data_test):
    """Tester si les doublons dans les données de test sont absents"""
    duplicates = data_test[data_test.duplicated()]
    assert duplicates.empty

def test_train_target_col(data_train):
    """Vérifier que la colonne 'TARGET' est présente dans les données d'entraînement"""
    assert 'TARGET' in data_train.columns

def test_train_test_sizes(data_train, data_test):
    """Vérifier que les données d'entraînement et de test ont le même nombre de colonnes (hors colonne 'TARGET')"""
    train_columns = data_train.drop(columns='TARGET').shape[1]
    test_columns = data_test.shape[1]
    assert train_columns == test_columns, f"Train columns: {train_columns}, Test columns: {test_columns}"
