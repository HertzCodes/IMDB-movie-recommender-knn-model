import json
import re
import string

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def text_cleaner(entry_title, entry_summary):
    if not nltk.corpus.stopwords.words("english"):
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    with open('datas/movies_infos_raw.json', 'r') as raw_movies:
        raw_movies = json.load(raw_movies)
        raw_movies[entry_title] = entry_summary
        stopwords = nltk.corpus.stopwords.words("english")
        url_pattern = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"

        for movie in raw_movies:
            raw_movies[movie] = re.sub(url_pattern, '', raw_movies[movie])
            raw_movies[movie] = nltk.word_tokenize(raw_movies[movie])
            raw_movies[movie] = [word for word in raw_movies[movie] if word not in string.punctuation]
            raw_movies[movie] = [word.lower() for word in raw_movies[movie]]
            raw_movies[movie] = [word for word in raw_movies[movie] if word not in stopwords]
            fdist = nltk.FreqDist(raw_movies[movie])
            raw_movies[movie] = [word for word in raw_movies[movie] if fdist[word] < fdist.N() * 0.1]
        with open('datas/movies_infos_clean_nltk.json', 'w') as cleaned_movies:
            cleaned_movies.write(json.dumps(raw_movies))


def vectorizer(entry_movie):
    with open('datas/movies_infos_clean_nltk.json', 'r') as datas:
        datas = json.load(datas)
        movies_vectors = {}
        for i in datas.keys():
            vectorize = TfidfVectorizer()
            tf_idf_scores = vectorize.fit_transform(datas[i])
            tf_idf_scores.toarray()
            print(len(tf_idf_scores))
            movies_vectors[i] = tf_idf_scores
        knn_model(movies_vectors, entry_movie)


def knn_model(movies_vectors, entry_movie):
    scores = {}
    for i in movies_vectors.keys():
        scores[i] = cosine_similarity(movies_vectors[i], movies_vectors[entry_movie])
    print(scores)


def start_process(entry_title, entry_summary):
    text_cleaner(entry_title, entry_summary)
    # vectorizer(entry_title)


