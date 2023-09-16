import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Set page configuration and title
st.set_page_config(
    page_title="Space Journey Prediction App",
    page_icon="🚀",
    layout="wide"
)

# Load custom CSS for styling
with open("styles.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


# Title and description
st.markdown("""
# Space Journey Prediction App
*Predict whether passengers will be transported on a space journey.*
""")

# Sidebar header
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

# Prediction header
st.markdown('<hr style="border:2px solid #007BFF">', unsafe_allow_html=True)
st.markdown('<p style="font-size: 24px; font-weight: bold;">Prediction Results</p>', unsafe_allow_html=True)

# Display prediction and prediction probability
prediction_text = 'Will be transported on a space journey' if prediction[0] == 1 else 'Will not be transported on a space journey'
st.write(f"Prediction: {prediction_text}")

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Add a space-themed image
image = Image.open("space_image.jpg")
st.image(image, use_column_width=True)
