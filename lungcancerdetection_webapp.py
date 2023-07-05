# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:00:19 2023

@author: ajith
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler


loaded_model = pickle.load(open('C:/Users/ajith/Downloads/Lung Cancer Detection/lungcancer_trained_model.sav', 'rb'))

#Creating a function for prediction

def lung_cancer_detection(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)

    #reshaping the array as we are predicting only for one instance

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    #standardize the input data

    sc = StandardScaler()
    std_data = sc.fit_transform(input_data_reshaped)
    print(std_data)


    prediction = loaded_model.predict(std_data)
    print(prediction)

    if(prediction[0] == 2):
        return "Lung Cancer Detected"
    else:
        return "You are safe"
    
    
def main():
    
    def add_bg_from_url():
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("https://www.statnews.com/wp-content/uploads/2023/01/AdobeStock_562452567-768x432.jpeg");
                    background-attachment: fixed;
                    background-size: cover
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
    
    add_bg_from_url()
    
    st.title('Lung Cancer Detection')
    st.write('*Created by Ajith Muraleedharan*')
    st.write("PS: This detection has been tested with 90% accuracy. Only numerical values are accepted.")
    
    
    #Input data from users
    
    GENDER = st.text_input('Gender (1 = Female, 2 = Male)')
    AGE = st.text_input('Age of the person')
    SMOKING = st.text_input('Smoking? (1 = No, 2 = Yes)')
    YELLOW_FINGERS = st.text_input('Yellow Fingers? (1 = No, 2 = Yes)')
    ANXIETY = st.text_input('Anxiety? (1 = No, 2 = Yes)')
    PEER_PRESSURE = st.text_input('Peer Pressure? (1 = No, 2 = Yes)')
    CHRONIC_DISEASE = st.text_input('Chronic Disease? (1 = No, 2 = Yes)')
    FATIGUE = st.text_input('Fatigue? (1 = No, 2 = Yes)')
    ALLERGY = st.text_input('Allergy? (1 = No, 2 = Yes)')
    WHEEZING = st.text_input('Wheezing? (1 = No, 2 = Yes)')
    ALCOHOL_CONSUMING = st.text_input('Alcohol Consuming? (1 = No, 2 = Yes)')
    COUGHING = st.text_input('Coughing? (1 = No, 2 = Yes)')
    SHORTNESS_OF_BREATH = st.text_input('Shortness of Breath? (1 = No, 2 = Yes)')
    SWALLOWING_DIFFICULTY = st.text_input('Swallowing Difficulty? (1 = No, 2 = Yes)')
    CHEST_PAIN = st.text_input('Chest Pain? (1 = No, 2 = Yes)')
    
    #Code for prediction
    
    lungcancer = ''
    
    #Creating a button
    
    if st.button("Lung Cancer Test Result"):
        
        lungcancer = lung_cancer_detection([GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN])
        
    st.success(lungcancer)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    