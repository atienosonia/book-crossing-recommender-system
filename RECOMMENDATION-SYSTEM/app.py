# Flask is the overall web framework

from flask import Flask, request, jsonify, render_template
import json

# joblib is used to unpickle the model

import joblib

# json is used to prepare the result
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import json

# importing pickle

import pickle

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

import numpy as np
# create new flask app here


app = Flask(__name__)

def load_model_data():
    model = pickle.load(open('model.pkl', 'rb'))
    book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))
    final_df = pickle.load(open('model_df.pkl', 'rb'))
    book_names = pickle.load(open('book_names.pkl', 'rb'))
    
    return model, book_pivot, final_df, book_names
model, book_pivot, final_df, book_names = load_model_data()



def recommend(book_name):
    # Find the index of the book in the book_pivot DataFrame
    book_id = np.where(book_pivot.index == book_name)[0][0]

    # Find the nearest neighbors and their distances for the given book
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1))

    # Iterate over the suggestions and create a list of book URLs with their names
    book_urls = []
    for idx in suggestions[0]:
        book_title = book_pivot.index[idx]
        url = final_df.loc[final_df['Book-Title'] == book_title, 'Image-URL-M'].values[0]
        book_urls.append((book_title, url))

    return book_urls


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        book_name = request.form['book_name']
        if not book_name:
            return jsonify({"error": "Invalid request"}), 400
        result = recommend(book_name)
        return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)