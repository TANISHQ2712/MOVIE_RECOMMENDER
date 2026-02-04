import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")

movies = movies[['title','overview']]
movies.dropna(inplace=True)

movies = movies.head(2000)

cv = CountVectorizer(max_features=2000, stop_words='english')
vectors = cv.fit_transform(movies['overview']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie = movie.lower()

    movies['title_lower'] = movies['title'].str.lower()

    if movie not in movies['title_lower'].values:
        print("\nMovie not found. Try exact name from dataset.\n")
        return

    index = movies[movies['title_lower'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    print("\nRecommended Movies:\n")
    for i in movie_list:
        print(movies.iloc[i[0]].title)
movie_name = input("Enter movie name: ")
recommend(movie_name)
