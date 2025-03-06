import pytest
import pandas as pd

@pytest.fixture
def get_url():
    url = 'postgresql://nytimesdb_owner:npg_TdyxEgfn1B2q@ep-square-paper-a2oqnwfa-pooler.eu-central-1.aws.neon.tech/nytimesdb?sslmode=require'
    return url

@pytest.fixture
def mock_data():
    # Crée un DataFrame factice pour les tests
    return pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10),
        'text': ['texte exemple'] * 10,
        # Ajoute les autres colonnes nécessaires
    })