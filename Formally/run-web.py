from flask import Flask
from flask import render_template, request

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

app = Flask(__name__)

deneme = ""

@app.route('/')
def index():
    return render_template("main_page.html")

@app.route('/handle_data', methods=['POST'])
def handle_data():

    normalized_documents = [request.form["user_input"]]

    features = []

    # open file and read the content in a list
    with open('features.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]

            # add item to the list
            features.append(currentPlace)

    #normalized_documents = ["naber askim"]

    from sklearn.feature_extraction.text import CountVectorizer

    Tfidf_Vector = pickle.load(open("feature(1).pkl", 'rb'))

    from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
    # = TfidfVectorizer(min_df = 0., max_df = 1., use_idf = True)
    Tfidf_Matrix = Tfidf_Vector.transform(normalized_documents)
    print(Tfidf_Matrix)
    Tfidf_df = pd.DataFrame(np.round(Tfidf_Matrix.todense(), 3), columns=features)

    x = Tfidf_df.values

    loaded_model = pickle.load(open("model.sav", 'rb'))



    y_pred = loaded_model.predict(x)

    if y_pred  == 1:
        y_pred = "Girdiginiz yazi resmidir (formal)."
    else:
        y_pred = "Girdiginiz yazi resmi degildir (informal)."

    return render_template("main_page.html", testText = y_pred,ttesx=request.form["user_input"])

if __name__ == '__main__':

    app.run()
