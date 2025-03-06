"""Streamlit app for visualizing New York Times business articles."""

import os
import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from wordcloud import WordCloud
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
NEONDB_URI = os.getenv("NEONDB_URI")

# Streamlit app
st.set_page_config(page_title="NYT Business News", page_icon="📰", layout="wide")
st.title("📰 New York Times - Business News Explorer")
st.markdown("Visualisation et analyse des articles récupérés par l'ETL.")

# Load data from the database
@st.cache_data
def load_data():
    """Load data from the database."""
    conn = psycopg2.connect(NEONDB_URI)
    query = """
        SELECT title, abstract, published_date
        FROM nyt_business_articles
        ORDER BY published_date DESC
        """
    data = pd.read_sql(query, conn)
    conn.close()

    data["published_date"] = pd.to_datetime(data["published_date"])
    return data


df = load_data()


# Sidebar filters
st.sidebar.header("📌 Filtres")
start_date, end_date = st.sidebar.date_input(
    "📅 Sélectionner une plage de dates",
    [df["published_date"].min(), df["published_date"].max()],
)

# Filter data based on user input
df_filtered = df[
    (df["published_date"] >= pd.Timestamp(start_date))
    & (df["published_date"] <= pd.Timestamp(end_date))
]

# Keyword search
search_query = st.sidebar.text_input("🔍 Recherche par mot-clé (titre ou abstract)")
if search_query:
    df_filtered = df_filtered[
        df_filtered["title"].str.contains(search_query, case=False, na=False)
        | df_filtered["abstract"].str.contains(search_query, case=False, na=False)
    ]

# Display filtered data
st.subheader("📋 Liste des articles récents")
st.dataframe(
    df_filtered[["published_date", "title", "abstract"]].reset_index(drop=True)
)

st.divider()

# Visualizations
st.subheader("📊 Nombre d'articles publiés par jour")
df_count = (
    df_filtered.groupby(df_filtered["published_date"].dt.date)
    .size()
    .reset_index(name="Nombre d'articles")
)
fig = px.bar(
    df_count,
    x="published_date",
    y="Nombre d'articles",
    title="Nombre d'articles publiés par jour",
    labels={"published_date": "Date", "Nombre d'articles": "Nombre d'articles"},
    color="Nombre d'articles",
    color_continuous_scale="blues",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Word cloud of keywords in titles
st.subheader("📌 Mots-clés les plus fréquents dans les titres")
TEXT_TITLE = " ".join(df_filtered["title"].dropna())
wordcloud = WordCloud(width=600, height=300, background_color="white").generate(TEXT_TITLE)
st.image(wordcloud.to_array(), use_container_width=False, width=600)


# Footer
st.markdown(
    """
    ---
    Made with ❤️ by [Christophe NORET](
    [GitHub](https://github.com/cnoret)
    [LinkedIn](https://www.linkedin.com/in/your-profile/)
    """
)