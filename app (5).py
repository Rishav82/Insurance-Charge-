# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd

# # Load the models
# lr = pickle.load(open("D:\\PYTHON1\\Data Science Project 1\\lr.pkl", 'rb')) 
# dt = pickle.load(open("D:\\PYTHON1\\Data Science Project 1\\dt.pkl", 'rb'))
# rf = pickle.load(open("D:\\PYTHON1\\Data Science Project 1\\rf.pkl", 'rb'))

# # Streamlit app setup
# st.title("Insurance Charge Prediction App")
# st.header('Fill the Details to generate the Predicted Insurance Charges')

# # Model selection
# options = st.sidebar.selectbox('Select ML Model', ['Linear Regression', 'Decision Tree Regression', 'Random Forest Regression'])

# # User input
# age = st.slider('Age', 18, 64)
# sex = st.selectbox('Sex', ['Male', 'Female'])
# bmi = st.slider('BMI', 6, 53)
# children = st.selectbox('Children', [0, 1, 2, 3, 4, 5])
# smoker = st.selectbox('Smoker', ['Yes', 'No'])
# region = st.selectbox('Region', ['North-West', 'South-East', 'South-West', 'North-East'])

# # Prediction button
# if st.button('Predict'):
#     # Convert categorical variables to numerical and one-hot encoding
#     sex = 1 if sex == 'Male' else 0
#     smoker = 1 if smoker == 'Yes' else 0
    
#     # One-hot encode region (3 binary features for 4 categories)
#     region_northwest = 1 if region == 'North-West' else 0
#     region_northeast = 1 if region == 'North-East' else 0
#     region_southeast = 1 if region == 'South-East' else 0

#     # Create the test array with the correct number of features
#     test = np.array([age, sex, bmi, children, smoker, region_northwest, region_northeast, region_southeast])
#     test = test.reshape(1, -1)  # Use -1 to infer the correct number of columns

#     # Predict using the selected model
#     if options == 'Linear Regression':
#         prediction = lr.predict(test)[0]
#     elif options == 'Decision Tree Regression':
#         prediction = dt.predict(test)[0]
#     else:
#         prediction = rf.predict(test)[0]

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the models
lr = pickle.load(open("D:\\PYTHON1\\Data Science Project 1\\lr.pkl", 'rb'))
dt = pickle.load(open("D:\\PYTHON1\\Data Science Project 1\\dt.pkl", 'rb'))
rf = pickle.load(open("D:\\PYTHON1\\Data Science Project 1\\rf.pkl", 'rb'))

# Streamlit app setup
st.title("Insurance Charge Prediction App")
st.header('Fill the Details to generate the Predicted Insurance Charges')

# Model selection
options = st.sidebar.selectbox('Select ML Model', ['Linear Regression', 'Decision Tree Regression', 'Random Forest Regression'])

# User input
age = st.slider('Age', 18, 64)
sex = st.selectbox('Sex', ['Male', 'Female'])
bmi = st.slider('BMI', 6, 53)
children = st.selectbox('Children', [0, 1, 2, 3, 4, 5])
smoker = st.selectbox('Smoker', ['Yes', 'No'])
region = st.selectbox('Region', ['North-West', 'South-East', 'South-West', 'North-East'])

# Prediction button
if st.button('Predict'):
    # Convert categorical variables to numerical and one-hot encoding
    sex = 1 if sex == 'Male' else 0
    smoker = 1 if smoker == 'Yes' else 0
    
    # One-hot encode region (3 binary features for 4 categories)
    region_northwest = 1 if region == 'North-West' else 0
    region_northeast = 1 if region == 'North-East' else 0
    region_southeast = 1 if region == 'South-East' else 0

    # Create the test array with the correct number of features
    test = np.array([age, sex, bmi, children, smoker, region_northwest, region_northeast, region_southeast])
    test = test.reshape(1, -1)  # Use -1 to infer the correct number of columns

    # Predict using the selected model
    if options == 'Linear Regression':
        prediction = lr.predict(test)[0]
    elif options == 'Decision Tree Regression':
        prediction = dt.predict(test)[0]
    else:
        prediction = rf.predict(test)[0]




    # Display the prediction
    st.success(f'The predicted insurance charge is: {prediction}')
    










