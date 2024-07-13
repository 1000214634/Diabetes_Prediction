# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 23:05:30 2024

@author: Ahmed
"""

import numpy as np
import pickle
import streamlit as st


#loading ther saved model
loaded_model=pickle.load(open('E:/proj/1)HEALTHCARE DOMAIN/3_PYTHON PROJ/trained_model.sav','rb'))


#Creating a function for prediction

def diabetes_prediction(input_data):
    
    #changing the input data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
        return 'The person is not diabetic'
    else:
        return'The person is diabetic'
        
        
        
def main():
    
    #giving a title
    
    st.title("Diabetes Prediction Web App")
   
  
    #getting the input data from the user
     
    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1)
    Glucose = st.number_input('Glucose Level', min_value=0, max_value=200, step=1)
    BloodPressure = st.number_input('Blood Pressure value', min_value=0, max_value=180, step=1)
    SkinThickness = st.number_input('Skin Thickness value', min_value=0, max_value=100, step=1)
    Insulin = st.number_input('Insulin Level', min_value=0, max_value=900, step=1)
    BMI = st.number_input('BMI value', min_value=0.0, max_value=70.0, step=0.1, format="%.1f")
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=2.5, step=0.01, format="%.2f")
    Age = st.number_input('Age of the Person', min_value=0, max_value=120, step=1)
    
    
    #code for prediction
    diagnosis=" "
    
    #creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
              
    
    
        st.success(diagnosis)
        
        
if __name__ == '__main__':
    main()