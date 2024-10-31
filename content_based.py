import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carica i dataset
movies = pd.read_csv("cleaned_movies.csv")
ratings = pd.read_csv("cleaned_ratings.csv")

# Calcola il punteggio medio di ciascun film
average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
average_ratings.columns = ['movieId', 'average_rating']

# Unisci i dati delle valutazioni con il dataset dei film
movies = pd.merge(movies, average_ratings, on='movieId')

# Trasforma i generi in una matrice TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calcola la similarità coseno tra i film
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Funzione per raccomandare film
def get_content_recommendations(title, cosine_sim=cosine_sim):
    # Controlla se il titolo è presente
    if title not in movies['title'].values:
        return f"Film '{title}' non trovato nel database."

    # Trova l'indice del film
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordina i film in base alla similarità
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Prendi i primi 10 film simili

    # Ottieni i titoli e altre informazioni
    movie_indices = [i[0] for i in sim_scores]
    
    # Crea un DataFrame per visualizzare i risultati
    recommendations = movies.iloc[movie_indices][['title', 'genres', 'average_rating']]

    # Ritorna le raccomandazioni formattate
    output = "Film simili a '{}':\n\n".format(title)
    for idx, row in recommendations.iterrows():
        output += "• {} (Rating: {:.1f}), Genere: {}\n".format(row['title'], row['average_rating'], row['genres'])
    
    return output

# Test
print(get_content_recommendations('Toy Story (1995)'))
