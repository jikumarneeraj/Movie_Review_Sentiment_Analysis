# step 1 -> Importing all the important libraries and load the model

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# load the IMDB dataset word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

# loadd the pre-trained model with ReLU activation function
model=load_model('simple_rnn_imdb.h5')
# used in the local machine
# model = load_model("N:/PW/udamy/DL_Projects/End to End deep learning project with simple RNN/code/simple_rnn_imdb.h5")

# step 2-> helper functions
# function to decode reviews
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# function to preprocess user input
def preprocess_text(text):
    vocab_size = 10000
    words=text.lower().split()
    # encoded_review=[word_index.get(word,2)+3 for word in words]
    encoded_review = [word_index.get(word, 2) + 3 for word in words if word_index.get(word, 0) < vocab_size]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

# now creating streamlit app
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

# taking input from the user
user_input=st.text_area('Movie review')

if st.button('classify'):
    preprocess_input=preprocess_text(user_input)

    # make prediction
    predication=model.predict(preprocess_input)
    sentiment='positive' if predication[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {predication[0][0]}')
else:
    st.write("Please enter a movie review.")