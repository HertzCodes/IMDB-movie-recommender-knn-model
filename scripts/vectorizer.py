import json
import re
import time

import numpy as np


def text_cleaner(entry_title, entry_summary):
    with open('datas/movies_infos_raw.json', 'r') as raw_datas:
        raw_datas = json.load(raw_datas)
        raw_datas[entry_title] = entry_summary
        stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                      'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
                      'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
                      'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                      'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                      'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                      'doing',
                      'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
                      'for',
                      'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                      'below', 'to',
                      'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
                      'once',
                      'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                      'most',
                      'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                      's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
                      'o', 're', 've', 'y',
                      'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
                      'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                      "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                      'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "one", "two", "three",
                      "four", "five", "six", "seven", "eight", "nine", "ten", "tn", ' ']
        for i in raw_datas:
            raw_datas[i] = raw_datas[i].lower()
            raw_datas[i] = re.sub("s/([()])//g", '', raw_datas[i])
            raw_datas[i] = re.sub(r'\W', ' ', raw_datas[i])
            raw_datas[i] = re.sub("[0-9]+", '', raw_datas[i])
            raw_datas[i] = raw_datas[i].split()
            for j in stop_words:
                while j in raw_datas[i]:
                    raw_datas[i].remove(j)
        cleaned_datas = json.dumps(raw_datas)
        with open('datas/movies_infos_cleaned.json', 'w') as file:
            file.write(cleaned_datas)


def tf_calculator(movie, term):
    return movie.count(term) / len(movie)


def idf_calculator(data, term):
    doc_freq = 0
    for i in data.keys():
        if term in data[i]:
            doc_freq += 1

    return np.log10(251 / doc_freq)


def vectorizer(data, movie, vectors_dict):
    for i in data[movie]:
        if vectors_dict[movie][i] == 0:
            tf_idf = idf_calculator(data, i) * tf_calculator(data[movie], i)
            vectors_dict[movie][i] = tf_idf


def start_process(entry_title, entry_summary):
    text_cleaner(entry_title, entry_summary)
    start = time.time()
    with open('datas/movies_infos_cleaned.json', 'r') as datas:
        datas = json.load(datas)
        vectors_dict = {}
        for i in datas.keys():
            vectors_dict[i] = {}
            for j in datas.values():
                for term in j:
                    vectors_dict[i][term] = 0
        for i in datas.keys():
            print(f'\rElapsed time: {int(time.time() - start)}s', end='')
            vectorizer(datas, i, vectors_dict)
        for i in vectors_dict.keys():  # CHANGES DICTS TO NUMPY ARRAYS
            vector = []
            for j in vectors_dict[i].keys():
                vector.append(vectors_dict[i][j])
            vector = np.array(vector)
            vectors_dict[i] = vector

        with open('datas/movie_vectors.json', 'w') as vectors_file:
            vectors_file.write(json.dumps(vectors_dict, cls=NumpyArrayEncoder))


class NumpyArrayEncoder(json.JSONEncoder):  # REFACTORS JSON ENCODERS TO SAVE NDARRAY
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
