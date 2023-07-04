# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:52:56 2023

@author: ajith
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

loaded_model = pickle.load(open('C:/Users/ajith/Downloads/lungcancer_trained_model.sav', 'rb'))
                                
input_data = (1,51,2,2,2,2,1,2,2,1,1,1,2,2,1)

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
    print("Lung Cancer Detected")
else:
    print("You are safe")                          