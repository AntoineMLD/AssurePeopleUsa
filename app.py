import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pickle
from preprocess_data import categorize_bmi

with open('exportModel.pkl', 'rb') as file:
    model = pickle.load(file)

# Interface utilisateur Streamlit pour la prédiction
st.sidebar.header('Entrez les informations pour estimer les charges d\'assurance')

# Champs de saisie pour les caractéristiques
age = st.sidebar.number_input('Âge', min_value=0, max_value=100, value=30)
sex = st.sidebar.radio('Sexe', ['male', 'female'])
bmi = st.sidebar.number_input('Indice de masse corporelle (BMI)', min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.number_input('Nombre d\'enfants', min_value=0, max_value=5, value=0)
smoker = st.sidebar.radio('Fumeur', ['yes', 'no'])
region = st.sidebar.selectbox('Région', ['southwest', 'southeast', 'northwest', 'northeast'])
# Préparation des données pour la prédiction



input_data = pd.DataFrame({'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children],
                           'smoker': [smoker], 'region': [region]})


y_pred_input = model.predict(input_data)

with open('exportR2.pkl', 'rb') as file:
    r2 = pickle.load(file)
# Affichage du score R2 en pourcentage
r2_percentage = r2 * 100
st.write(f"Taux de prédibilité : {r2_percentage:.2f}%")

# Arrondir et afficher la prédiction à deux chiffres après la virgule
rounded_prediction = round(y_pred_input[0], 2)
st.write(f"<span style='font-size:24px'>Estimation des charges d\'assurance :</span> <span style='color:red;font-size:24px'>{rounded_prediction:.2f}</span><span style='font-size:24px'> $</span>", unsafe_allow_html=True)


