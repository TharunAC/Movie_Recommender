import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import glob

def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=fe90c7a0ed2590d41538669c534b8880')
    data = response.json()
    return "https://image.tmdb.org/t/p/w500"+data['poster_path']

def recommend(selected_movie):
    movie_index = movies[movies['title'] == selected_movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]

    recommend_movies = []
    recommend_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        poster_path = fetch_poster(movie_id)
        if poster_path:
            recommend_movies.append(movies.iloc[i[0]].title)
            recommend_posters.append(poster_path)
    return recommend_movies, recommend_posters




st.title('Movie Recommender System')

# Load movie data
movies_list = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_list)

chunks = []
chunk_files = glob.glob('chunk_*.pkl')
for chunk_file in chunk_files:
    # Load the chunk
    chunk = pd.read_pickle(chunk_file)

    # Check if the loaded chunk is a numpy array
    if isinstance(chunk, np.ndarray):
        chunks.append(chunk)
    else:
        print(f"File {chunk_file} did not load as a numpy array. Instead loaded as {type(chunk)}.")

# Concatenate the numpy arrays if all chunks are valid
if chunks:
    combined_array = np.concatenate(chunks, axis=0)
    # Convert the combined numpy array to a pandas DataFrame
    similarity = pd.DataFrame(combined_array)
    #print("Concatenation successful.")
    #print(similarity)
else:
    print("No valid numpy array chunks to concatenate.")

# Select movie
selected_movie = st.selectbox(
    "Select a movie to get recommendations:",
    movies['title'].values
)
if st.button("Recommend"):
    names, posters = recommend(selected_movie)
    num_cols = 5  # Number of columns for movie recommendations

    num_movies = len(names)  # Total number of movie recommendations

    # Calculate number of rows needed to display all movie recommendations
    num_rows = (num_movies + num_cols - 1) // num_cols

    # Create columns for each movie recommendation
    col1, col2, col3, col4, col5 = st.columns(num_cols)

    index = 0  # Initialize index to track movie recommendations
    for row in range(num_rows):
        # Display movie recommendations in each column
        if index < num_movies:
            col1.markdown(names[index])
            col1.image(posters[index], use_column_width=True)
            index += 1
        if index < num_movies:
            col2.markdown(names[index])
            col2.image(posters[index], use_column_width=True)
            index += 1
        if index < num_movies:
            col3.markdown(names[index])
            col3.image(posters[index], use_column_width=True)
            index += 1
        if index < num_movies:
            col4.markdown(names[index])
            col4.image(posters[index], use_column_width=True)
            index += 1
        if index < num_movies:
            col5.markdown(names[index])
            col5.image(posters[index], use_column_width=True)
            index += 1

filtered_df=[]
def recommend3(actor_name,actor_list):
    filtered_df = actor_list[actor_list['Actor'] == actor_name]
    filtered_df=filtered_df.sort_values(by='Rating', ascending=False)[:20]
    filtered_df = filtered_df[['movie_id', 'title']]
    recommend_actor_movies=[]
    recommend_actor_title=[]
    for i,row in filtered_df.iterrows():
        movie_id = row['movie_id']
        poster_path = fetch_poster(movie_id)
        if poster_path:
            recommend_actor_movies.append(row['title'])
            recommend_actor_title.append(poster_path)
    return recommend_actor_movies, recommend_actor_title


actor_list=pickle.load(open('actor_dict.pkl','rb'))
actor_list=pd.DataFrame(actor_list)
st.header('Actor Recommendations')
actor_name = st.text_input("Enter an actor's name:")
if st.button("Recommedations"):
    names, posters = recommend3(actor_name,actor_list)
    num_cols = 5  # Number of columns for movie recommendations

    num_movies = len(names)  # Total number of movie recommendations

    # Calculate number of rows needed to display all movie recommendations
    num_rows = (num_movies + num_cols - 1) // num_cols

    # Create columns for each movie recommendation
    columns = [st.columns(num_cols) for _ in range(num_rows)]

    index = 0  # Initialize index to track movie recommendations
    for row in range(num_rows):
        for col in range(num_cols):
            if index < num_movies:
                columns[row][col].markdown(names[index])
                columns[row][col].image(posters[index], use_column_width=True)
                index += 1

