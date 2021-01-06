import streamlit as st

title = st.title('Sentiment Analysis from IMDB dataset')

header= st.header('Example of the test set')

Models = st.sidebar.selectbox(
    'Choose the model',
    ('RNN', 'LSTM','Bert'))


import re
import joblib

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


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)



logicregression = joblib.load('final_model.pkl')
#for i in range(10):

st.subheader('Movie comment')


st.text_area("", value=reviews_test[20050], height=350)


st.subheader('prediction')

if logicregression.predict(X_test[20050]) == 0:
    prediction = 'Negative'
else:
    prediction = 'Positive'

target = [1 if i < 12500 else 0 for i in range(25050)]

st.text(prediction)

st.subheader('label')

st.text(target[20050])

st.subheader('Accuracy')


from sklearn.metrics import accuracy_score


accuracy = accuracy_score(target, logicregression.predict(X_test))

st.write(accuracy)