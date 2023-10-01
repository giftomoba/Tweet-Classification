import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd

#load the trained model:
with open(r'C:\Users\ohiom\Desktop\MLOps Class\Project 1\Model\rand_class.pkl', 'rb') as rand_class:
    model = pickle.load(rand_class)

with open(r'C:\Users\ohiom\Desktop\MLOps Class\Project 1\Model\tf_idVect.pkl', 'rb') as vect:
    vectorizer = pickle.load(vect)

# Create title
st.write('## Text Sentiment Analysis App')

# Create a text input widget for user input
user_input = st.text_area('Enter Your Text Here:')
  
# Function to lemmatize and vectorize the text input:
def vectorizer_func(text, vectorizer):
    words = WordNetLemmatizer().lemmatize(text)
    text_list = []
    text_list.append(words)
    data_ = pd.DataFrame(text_list, columns=['text'])
    vectorized_text = vectorizer.transform(data_['text']).toarray()
    return vectorized_text

# Create a button to trigger the sentiment analysis:
if st.button('Predict'):
    vectorized_input = vectorizer_func(user_input, vectorizer)
    # Make Prediction using the loaded model:
    prediction = model.predict(vectorized_input)
    if prediction[0] == 0:
        sentiment = 'Negative'
    elif prediction[0] == 1:
        sentiment = 'Neutral'
    else:
        sentiment = 'Positive'
    st.write(prediction[0])

    # Display the sentiment:
    st.write(f'## Sentiment is: {sentiment}')
