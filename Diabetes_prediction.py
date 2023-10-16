from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

st.set_page_config(layout='wide')
st.title('Diabetes Prediction')


df = pd.read_csv('D:\Data_Excel\diabetes_prediction_dataset.csv')

eng = OrdinalEncoder()

df['gender'] = eng.fit_transform(df[['gender']])
df['smoking_history'] = eng.fit_transform(df[['smoking_history']])


y = df['diabetes']
x = df.drop(['diabetes'], axis = 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)


model = DecisionTreeClassifier()
result = model.fit(x_train, y_train)

col1, col2= st.columns(2)

with col1:

    #gender
    Gr=st.selectbox('Gender',('Male','Female','Others'))
    if Gr=='Male':
        gen=1
    elif Gr=='Female':
        gen=0
    else:
        gen=2

    #age
    age = st.number_input('Age',format='%g')  

    #hypertension
    Hyp=st.selectbox('Hypertension',('Yes','No'))
    if Hyp=='Yes':
        Hyt=1
    else :
        Hyt=0

    #Heart disease
    Htd=st.selectbox('Heart Disease',('Yes','No'))
    if Htd=='Yes':
        Hd=1
    else :
        Hd=0

with col2:

    #smoking_history
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


    #bmi
    bmi = st.number_input('BMI')  


    HbA1c = st.number_input('HbA1c  :blue[( 3 months - Avg Blood Glucose level )]')  


    bg = st.number_input('Blood Glucose Level')  

detail=[gen,age,Hyt,Hd,Sh,bmi,HbA1c,bg]


users = np.array([detail])

users_predict = result.predict(users)

if st.button('Calculate'):
    if users_predict == 0:
      st.success("You are Diabetes free, Enjoy :smile:")
      st.snow()

    else:
         st.success("You have the chances of having the Diabetes. :red[ Please consult the Doctor]")  
         