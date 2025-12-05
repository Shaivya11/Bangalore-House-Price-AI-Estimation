import pickle
import json
import streamlit as st
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)


def get_location_names():
    with open(os.path.join(dir_path, 'columns.json'), 'rb') as f:
        return json.load(f)["data_columns"]
    
def load_ML_model():
    with open(os.path.join(dir_path, 'bangalore_home_price_model.pickle'), 'rb') as f:
        return pickle.load(f)
    
def get_estimated_price(location,sqft, bath, bhk):
    try:
        loc_index = locations.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(locations))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >=0:
        x[loc_index] = 1

    return round(model.predict([x])[0])

       
locations = get_location_names()
model = load_ML_model()


st.title("Bangalore House Price Estimator")

background_image ="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://img.freepik.com/premium-photo/blur-images-modern-private-house-blurred-house-design-backgroud_861973-34512.jpg");
  background-size: cover;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

sqft = st.number_input("Area (Sqft)",min_value=0, value=1000, step=100)
bhk = st.number_input("BHK",min_value=1, max_value=5, value=1, step=1)
bath = st.number_input("Bath",min_value=1, max_value=5, value=1, step=1)

location = st.selectbox("Location", options=locations[3:],placeholder="Select a location...")

if st.button("Estimate Price"):
    predicted_price = get_estimated_price(location,sqft, bath, bhk)
    if predicted_price:
        st.success("{} Lakhs".format(predicted_price))