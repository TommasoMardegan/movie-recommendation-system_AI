import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carica il dataset pulito
movies = pd.read_csv("cleaned_movies.csv")

# Trasforma i generi in una matrice TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calcola la similarità coseno tra i film
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Funzione per raccomandare film
def get_content_recommendations(title, cosine_sim=cosine_sim):
    # Trova l'indice del film
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Ordina i film in base alla similarità
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Prendi i primi 10 film simili
    
    # Ottieni i titoli
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Test
print("Film simili a 'Toy Story':\n", get_content_recommendations('Toy Story (1995)'))
