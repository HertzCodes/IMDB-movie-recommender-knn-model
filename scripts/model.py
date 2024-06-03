import heapq
import json

import numpy as np
from numpy.linalg import norm


def knn_model(entry_movie):
    with open('datas/movie_vectors.json', 'r') as vectors:
        vectors = json.load(vectors)
        calculate_cosine_similarity(vectors, entry_movie)


def calculate_cosine_similarity(vectors, entry_movie):
    similar_movies = {}
    for movie in vectors:
        similar_movies[movie] = (np.dot(vectors[movie], vectors[entry_movie]) / (
                norm(vectors[movie]) * norm(vectors[entry_movie])))
    print('OUR MODEL: ', heapq.nlargest(6, similar_movies, key=similar_movies.get)[1:])
