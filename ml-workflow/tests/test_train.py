# test_app_trial_ny_times_sa.py

import pytest
import pandas as pd
import numpy as np
from app.train import load_data, preprocess_data, apply_sentiment_analysis, preprocess_sentiment_scores, split_data_by_label, train_and_predict

@pytest.fixture
def get_url():
    url = 'postgresql://nytimesdb_owner:npg_TdyxEgfn1B2q@ep-square-paper-a2oqnwfa-pooler.eu-central-1.aws.neon.tech/nytimesdb?sslmode=require'
    return url

# Test pour vérifier que les données sont chargées correctement
def test_load_data(get_url):
    url = get_url
    df = load_data(url)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

# Test pour vérifier que le prétraitement des données fonctionne
def test_preprocess_data(get_url):
    url = get_url
    df = load_data(url)
    df_business = preprocess_data(df)
    assert isinstance(df_business, pd.DataFrame)
    assert 'text' in df_business.columns
    assert 'date' in df_business.columns

# Test pour vérifier que l'analyse de sentiment fonctionne
def test_apply_sentiment_analysis(get_url):
    url = get_url
    df = load_data(url)
    df_business = preprocess_data(df)
    df_SA = apply_sentiment_analysis(df_business)
    assert isinstance(df_SA, pd.DataFrame)
    assert 'label' in df_SA.columns
    assert 'score' in df_SA.columns

# Test pour vérifier que les scores de sentiment sont prétraités correctement
def test_preprocess_sentiment_scores(get_url):
    url = get_url
    df = load_data(url)
    df_business = preprocess_data(df)
    df_SA = apply_sentiment_analysis(df_business)
    df_pivot = preprocess_sentiment_scores(df_SA)
    assert isinstance(df_pivot, pd.DataFrame)
    assert not df_pivot.empty

# Test pour vérifier que les données sont divisées correctement par label
def test_split_data_by_label(get_url):
    url = get_url
    df = load_data(url)
    df_business = preprocess_data(df)
    df_SA = apply_sentiment_analysis(df_business)
    df_pivot = preprocess_sentiment_scores(df_SA)
    df_Very_Negative, df_Negative, df_Neutral, df_Positive, df_Very_Positive = split_data_by_label(df_pivot)
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
    forecast = train_and_predict(df, 'Test Label')
    assert isinstance(forecast, pd.DataFrame)
    assert 'yhat' in forecast.columns
    assert 'ds' in forecast.columns