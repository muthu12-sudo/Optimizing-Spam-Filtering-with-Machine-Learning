from flask import Flask, render_template, request
import pickle
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

load_model = load_model('spam.h5')
cv = pickle.load(open('cv1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/spam', methods=['POST','GET'])
def prediction():
  return render_template('spam.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method =='POST':
        message = request.form['message']
        data = message

        new_review = str(data)
        new_review = re.sub('[^a-zA-Z]', ' ', new_review)
        new_review = new_review.lower()
        new_review = new_review.split()
        ps = PorterStemmer()
        new_review = [ps.stem(word) for word in new_review]
        new_review = ' '.join(new_review)
        new_corpus = [new_review]
        new_X_test = cv.transform(new_corpus).toarray()
        new_y_pred = load_model.predict(new_X_test)
        new_X_pred = np.where(new_y_pred > 0.5, 1, 0)

        if new_X_pred[0][0] == 1:
            prediction = "Spam"
        else:
            prediction = "Not a Spam"

        return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=False)
