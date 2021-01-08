import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
import numpy as np

from numpy.random import randint 

import re
import joblib


title = st.title('Sentiment Analysis from IMDB dataset')

header= st.header('Example of the test set')


Models = st.sidebar.selectbox(
    'Choose the model',    ('Linear Regression', 'LSTM','Bert'))
st.sidebar.text('')
st.sidebar.markdown('[More resources and tools](https://github.com/epadam/machine-learning-overview/blob/master/NLP.md)')




def read_file():
    reviews_train = []
    for line in open('aclImdb/movie_data/full_train.txt', 'r'):
        reviews_train.append(line.strip())  
    
    reviews_test = []
    for line in open('aclImdb/movie_data/full_test.txt', 'r'):
        reviews_test.append(line.strip())
        
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    
    def preprocess_reviews(reviews):
        reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
        reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
        return reviews
        
    reviews_train_clean = preprocess_reviews(reviews_train)
    reviews_test_clean = preprocess_reviews(reviews_test)
    
    cv = CountVectorizer(binary=True)
    cv.fit(reviews_train_clean)
    X = cv.transform(reviews_train_clean)
    X_test = cv.transform(reviews_test_clean)

    return X, X_test, reviews_test


read_file()


logicregression = joblib.load('final_model.pkl')

ram = randint(0,25000)

st.subheader('Movie comment')


st.text_area("", value=reviews_test[ram], height=350)


st.subheader('prediction')

if logicregression.predict(X_test[ram]) == 0:
    prediction = 'Negative'
else:
    prediction = 'Positive'

target = [1 if i < 12500 else 0 for i in range(25000)]

st.text(prediction)

st.subheader('label')

st.text(target[ram])

st.subheader('Accuracy')





accuracy = accuracy_score(target, logicregression.predict(X_test))

st.write(accuracy)