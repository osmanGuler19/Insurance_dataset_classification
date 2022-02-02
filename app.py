import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
import pickle

st.title("Medical Insurance Risk Predict")

age = st.selectbox('Age',('<=30','30-50','50+',))
sex = st.selectbox('Sex',('Female','Male',))
bmi = st.selectbox('BMI',('<=18.5','18.5-25','25-29','30-35','35+'))
children = st.selectbox('Children',('0','1','2','3','4','5'))
smoker = st.selectbox('Smoker',('Yes','No'))
region = st.selectbox('Region',('Southwest','Southeast','Northwest','Northeast'))
def predict():
    model = pickle.load(open('random_forest_model.pkl', 'rb'))
    my_age=-1
    my_sex = -1
    my_bmi = -1
    my_children=-1
    my_smoker =-1
    my_region=-1
    if age == '<=30':
        my_age=0
    elif age == '30-50':
        my_age=1
    else:
        my_age=2
    if sex == '<=Female':
        my_sex=0
    else:
        my_sex=1
    if bmi=='<=18.5':
        my_bmi=0
    elif bmi == '18.5-25':
        my_bmi=1
    elif bmi == '25-29':
        my_bmi=2
    elif bmi == '30-35':
        my_bmi=3
    else:
        my_bmi=4

    if children=='0':
        my_children=0
    elif children=='1':
        my_children=1
    elif children=='2':
        my_children=2
    elif children=='3':
        my_children=3
    elif children=='4':
        my_children=4
    else:
        my_children = 5

    if smoker=='Yes':
        my_smoker=1
    else:
        my_smoker=2

    if region=='Southwest':
        my_region=0
    elif region=='Southeast':
        my_region=1
    elif region=='Northwest':
        my_region=2
    else:
        my_region = 3

    tahmin=model.predict([[my_sex,my_children,my_smoker,my_region,my_bmi,my_age]])
    tahmin_text=""
    if tahmin==0:
        tahmin_text="Low Risk"
    elif tahmin==1:
        tahmin_text= "Normal Risk"
    else:
        tahmin_text = "High Risk"

    st.write("Model Sonucu : {}".format(tahmin_text))

button = st.button('Predict', on_click=predict)

df = pd.read_csv('insurance.csv')
st.write(df)



st.write(model)