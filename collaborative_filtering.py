import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Carica i dataset
ratings = pd.read_csv("cleaned_ratings.csv")
movies = pd.read_csv("cleaned_movies.csv")

# Crea una matrice Utente-Film
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Calcola la similarità tra utenti
user_similarity = cosine_similarity(user_movie_matrix.fillna(0))
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Funzione per raccomandare film
def recommend_movies(user_id, num_recommendations=5):
    # Ottieni i rating dell'utente
    user_ratings = user_movie_matrix.loc[user_id].dropna()

    # Se l'utente non ha film valutati, restituisci un messaggio
    if user_ratings.empty:
        return f"L'utente {user_id} non ha valutato alcun film."

    # Trova utenti simili
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)

    # Se non ci sono utenti simili, restituisci un messaggio
    if similar_users.empty:
        return f"Nessun utente simile trovato per l'utente {user_id}."

    # Filtra la matrice per gli utenti simili
    similar_users_ratings = user_movie_matrix.loc[similar_users.index].fillna(0)

    # Calcola il rating pesato
    weighted_ratings = similar_users.dot(similar_users_ratings) / similar_users.sum()

    # Filtra i film già valutati
    recommendations = weighted_ratings.drop(user_ratings.index).nlargest(num_recommendations)

    # Controlla se ci sono raccomandazioni
    if recommendations.empty:
        return f"Nessuna raccomandazione disponibile per l'utente {user_id}. L'utente ha valutato tutti i film."

    # Unisci le raccomandazioni con il DataFrame dei film per ottenere dettagli
    recommended_movies = movies[movies['movieId'].isin(recommendations.index)][['title', 'year', 'genres']]

    # Aggiungi il rating raccomandato
    recommended_movies['predicted_rating'] = recommendations.values

    # Ritorna le raccomandazioni formattate
    output = f"Raccomandazioni per l'utente {user_id}:\n\n"
    for idx, row in recommended_movies.iterrows():
        output += f"• {row['title']} ({row['year']}), Genere: {row['genres']}, Predicted Rating: {row['predicted_rating']:.1f}\n"

    return output

# Test delle raccomandazioni
print(recommend_movies(1))
