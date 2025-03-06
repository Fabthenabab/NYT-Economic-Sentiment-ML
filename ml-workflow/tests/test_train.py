# test_app_trial_ny_times_sa.py

import pytest
import pandas as pd
import numpy as np
import app.train
from app.train import load_data, preprocess_data, apply_sentiment_analysis, preprocess_sentiment_scores, split_data_by_label, train_and_predict


# Utiliser le mock au lieu de la base de données
def test_load_data(monkeypatch, mock_data):
    # Remplacer la fonction load_data par une version qui retourne les données mock
    monkeypatch.setattr("app.train.load_data", lambda url: mock_data)
    
    url = "postgresql://..."
    df = app.train.load_data(url)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

# Test pour vérifier que le prétraitement des données fonctionne
def test_preprocess_data(monkeypatch, mock_data):
    # Remplacer la fonction load_data par une version qui retourne les données mock
    monkeypatch.setattr("app.train.load_data", lambda url: mock_data)
    
    url = "postgresql://..."
    df = app.train.load_data(url)
    df_business = app.train.preprocess_data(df)
    assert isinstance(df_business, pd.DataFrame)
    assert 'text' in df_business.columns
    assert 'date' in df_business.columns

# Test pour vérifier que l'analyse de sentiment fonctionne
def test_apply_sentiment_analysis(monkeypatch, mock_data):
    # Remplacer la fonction load_data par une version qui retourne les données mock
    monkeypatch.setattr("app.train.load_data", lambda url: mock_data)
    
    url = "postgresql://..."
    df = app.train.load_data(url)
    df_business = app.train.preprocess_data(df)
    df_SA = app.train.apply_sentiment_analysis(df_business)
    assert isinstance(df_SA, pd.DataFrame)
    assert 'label' in df_SA.columns
    assert 'score' in df_SA.columns

# Test pour vérifier que les scores de sentiment sont prétraités correctement
def test_preprocess_sentiment_scores(monkeypatch, mock_data):
    # Remplacer la fonction load_data par une version qui retourne les données mock
    monkeypatch.setattr("app.train.load_data", lambda url: mock_data)
    
    url = "postgresql://..."
    df = app.train.load_data(url)
    df_business = app.train.preprocess_data(df)
    df_SA = app.train.apply_sentiment_analysis(df_business)
    df_pivot = app.train.preprocess_sentiment_scores(df_SA)
    assert isinstance(df_pivot, pd.DataFrame)
    assert not df_pivot.empty

# Test pour vérifier que les données sont divisées correctement par label
def test_split_data_by_label(monkeypatch, mock_data):
    # Remplacer la fonction load_data par une version qui retourne les données mock
    monkeypatch.setattr("app.train.load_data", lambda url: mock_data)
    
    url = "postgresql://..."
    df = app.train.load_data(url)
    df_business = app.train.preprocess_data(df)
    df_SA = app.train.apply_sentiment_analysis(df_business)
    df_pivot = app.train.preprocess_sentiment_scores(df_SA)
    df_Very_Negative, df_Negative, df_Neutral, df_Positive, df_Very_Positive = app.train.split_data_by_label(df_pivot)
    assert isinstance(df_Very_Negative, pd.DataFrame)
    assert isinstance(df_Negative, pd.DataFrame)
    assert isinstance(df_Neutral, pd.DataFrame)
    assert isinstance(df_Positive, pd.DataFrame)
    assert isinstance(df_Very_Positive, pd.DataFrame)

# Test pour vérifier que le modèle Prophet fonctionne
def test_train_and_predict():
    df = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=100),
        'y': np.random.rand(100)
    })
    forecast = app.train.train_and_predict(df, 'Test Label')
    assert isinstance(forecast, pd.DataFrame)
    assert 'yhat' in forecast.columns
    assert 'ds' in forecast.columns