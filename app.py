import streamlit as st
import pandas as pd
from content_based import get_content_recommendations
from collaborative_filtering import recommend_movies

# Carica i dati
movies = pd.read_csv("cleaned_movies.csv")

st.title("Sistema di Raccomandazione di Film")

# Seleziona il tipo di raccomandazione
option = st.selectbox("Seleziona il tipo di raccomandazione:", ("Content-Based", "Collaborative Filtering"))

# Content-Based Filtering
if option == "Content-Based":
    title = st.selectbox("Seleziona un film:", movies['title'].values)
    if st.button("Raccomanda Film Simili"):
        recommendations = get_content_recommendations(title)
        st.write("Film raccomandati:")
        for movie in recommendations:
            st.write(movie)

# Collaborative Filtering
if option == "Collaborative Filtering":
    user_id = st.number_input("Inserisci l'ID utente:", min_value=1, step=1)
    if st.button("Raccomanda Film per Utente"):
        recommendations = recommend_movies(user_id)
        st.write("Film raccomandati per l'utente", user_id)
        for movie_id in recommendations.index:
            movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
            st.write(movie_title)
