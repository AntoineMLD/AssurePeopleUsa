import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pickle


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

# input_data = cleanerInput(input_data)
    #Crée une colone de smoker en fonction du BMI
input_data['smoker_binary'] = (input_data['smoker'] == 'yes').astype(int)

label_encoder = LabelEncoder()
columns_to_encode = ['sex', 'smoker']
for column in columns_to_encode:
    input_data[column] = label_encoder.fit_transform(input_data[column])  # Encodage des colonnes

# Créer des variables binaires avec des noms explicites pour la colonne 'region'
# region_dummies_named = pd.get_dummies(input_data['region'], prefix='is', prefix_sep='_')

region_dummies_named = pd.get_dummies(input_data['region'], prefix='is', prefix_sep='_')
# Colonnes à conserver dans le DataFrame final
desired_columns = ['is_northeast', 'is_northwest', 'is_southeast', 'is_southwest']
# Ajouter les colonnes manquantes avec des valeurs par défaut de 0
for col in desired_columns:
    if col not in region_dummies_named.columns:
        region_dummies_named[col] = 0
# Concaténer ces variables binaires avec le DataFrame original
input_data = pd.concat([input_data, region_dummies_named[desired_columns]], axis=1)

#Création des intervalles pour les catégories BMI
bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]  # Les limites des catégories

#Étiquettes pour les catégories BMI
labels = [
    'underweight', 'normal weight', 'overweight',
    'obesity class I', 'obesity class II', 'obesity class III'
    ]

#Utilisation de pd.cut pour créer de nouvelles colonnes basées sur les catégories BMI
input_data['BMI_category'] = pd.cut(input_data['bmi'], bins=bins, labels=labels, right=False)

#Utilisation de pd.get_dummies pour obtenir des colonnes binaires pour chaque catégorie
BMI_dummies = pd.get_dummies(input_data['BMI_category'])

#Ajout des colonnes binaires au DataFrame X
input_data = pd.concat([input_data, BMI_dummies], axis=1)

input_data['bmi_smoker'] = input_data['bmi'] * input_data['smoker_binary']
input_data = input_data.drop('smoker_binary', axis=1)

#Suppression de la colonne 'BMI_category' car elle n'est plus nécessaire
input_data = input_data.drop('BMI_category', axis=1)

y_pred_input = model.predict(input_data)

with open('exportR2.pkl', 'rb') as file:
    r2 = pickle.load(file)

# Affichage du score R2 en pourcentage
r2_percentage = r2 * 100
st.write(f"Taux de prédibilité : {r2_percentage:.2f}%")

# Arrondir et afficher la prédiction à deux chiffres après la virgule
rounded_prediction = round(y_pred_input[0], 2)
st.write(f"<span style='font-size:24px'>Estimation des charges d\'assurance :</span> <span style='color:red;font-size:24px'>{rounded_prediction:.2f}</span><span style='font-size:24px'> $</span>", unsafe_allow_html=True)
