import pandas as pd

# Carica i dataset
movies = pd.read_csv("movies.csv")      # Colonne: movieId, title, genres
ratings = pd.read_csv("ratings.csv")    # Colonne: userId, movieId, rating, timestamp

# Pulisci i dati (gestisci valori mancanti e formatta le colonne)
movies['genres'] = movies['genres'].fillna('')  # Sostituisce valori mancanti
movies['title'] = movies['title'].fillna('')    # Sostituisce valori mancanti

# Salva i dati puliti in nuovi file CSV
movies.to_csv("cleaned_movies.csv", index=False)
ratings.to_csv("cleaned_ratings.csv", index=False)

print("Dati preparati e salvati con successo.")
