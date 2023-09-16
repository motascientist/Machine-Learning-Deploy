import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Space Journey Prediction App

This app predicts whether passengers will be transported on a space journey!

""")

st.sidebar.header('User Input Features')

# Collects user input features into a dictionary
def user_input_features():
    HomePlanet = st.sidebar.selectbox('Home Planet', [0, 1])
    CryoSleep = st.sidebar.selectbox('Cryo Sleep', [0, 1])
    Destination = st.sidebar.selectbox('Destination', [0, 1])
    Age = st.sidebar.slider('Age', 0, 100, 30)
    VIP = st.sidebar.selectbox('VIP', [0, 1])
    RoomService = st.sidebar.slider('Room Service', 0.0, 5000.0, 0.0)
    FoodCourt = st.sidebar.slider('Food Court', 0.0, 5000.0, 0.0)
    ShoppingMall = st.sidebar.slider('Shopping Mall', 0.0, 5000.0, 0.0)
    Spa = st.sidebar.slider('Spa', 0.0, 5000.0, 0.0)
    VRDeck = st.sidebar.slider('VR Deck', 0.0, 5000.0, 0.0)
    
    data = {
        'HomePlanet': HomePlanet,
        'CryoSleep': CryoSleep,
        'Destination': Destination,
        'Age': Age,
        'VIP': VIP,
        'RoomService': RoomService,
        'FoodCourt': FoodCourt,
        'ShoppingMall': ShoppingMall,
        'Spa': Spa,
        'VRDeck': VRDeck
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Displays the user input features
st.subheader('User Input Features')
st.write(input_df)

# Reads in saved classification model
load_clf = pickle.load(open('space_titanic.pkl', 'rb'))

# Apply the model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

st.subheader('Prediction')
st.write('Will be transported on a space journey' if prediction[0] == 1 else 'Will not be transported on a space journey')

st.subheader('Prediction Probability')
st.write(prediction_proba)
