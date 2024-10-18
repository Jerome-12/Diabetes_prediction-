# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# Streamlit page 
st.set_page_config(layout='wide')
st.title('Diabetes Prediction')

# Load the csv file
df = pd.read_csv('F:\Data_Excel\diabetes_prediction_dataset.csv')

# Initialize the OrdinalEncoder
eng = OrdinalEncoder()

# Transform categorical features
df['gender'] = eng.fit_transform(df[['gender']])
df['smoking_history'] = eng.fit_transform(df[['smoking_history']])

# Define target and features
y = df['diabetes']
x = df.drop(['diabetes'], axis = 1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)

# Train the Decision Tree Classifier
model = DecisionTreeClassifier()
result = model.fit(x_train, y_train)

# Create two columns in Streamlit layout
col1, col2= st.columns(2)

with col1:

    #Gender selection
    Gr=st.selectbox('Gender',('Male','Female','Others'))
    if Gr=='Male':
        gen=1
    elif Gr=='Female':
        gen=0
    else:
        gen=2

    #Age input
    age = st.number_input('Age',format='%g')  

    #Hypertension selection
    Hyp=st.selectbox('Hypertension',('Yes','No'))
    if Hyp=='Yes':
        Hyt=1
    else :
        Hyt=0

    #Heart disease selection
    Htd=st.selectbox('Heart Disease',('Yes','No'))
    if Htd=='Yes':
        Hd=1
    else :
        Hd=0

with col2:

    #Smoking history selection
    Sht=st.selectbox('Smoking History',('Never','Current','Former','Ever','Not Current','Others'))
    if Sht=='Never':
        Sh=4
    elif Sht=='Current':
        Sh=1
    elif Sht=='Former':
        Sh=3
    elif Sht=='Ever':
        Sh=2
    elif Sht=='Not Current':
        Sh=5
    else :
        Sh=0


    #BMI input
    bmi = st.number_input('BMI')  

    #HbA1c input
    HbA1c = st.number_input('HbA1c  :blue[( 3 months - Avg Blood Glucose level )]')  

    # Blood Glucose Level input
    bg = st.number_input('Blood Glucose Level')  

# Prepare input for prediction
detail=[gen,age,Hyt,Hd,Sh,bmi,HbA1c,bg]
users = np.array([detail])

# Prediction
users_predict = result.predict(users)

if st.button('Calculate'):
    if users_predict == 0:
      st.success("You are Diabetes free, Enjoy :smile:")

    else:
         st.success("You have the chances of having the Diabetes. :red[ Please consult the Doctor]")  
         
