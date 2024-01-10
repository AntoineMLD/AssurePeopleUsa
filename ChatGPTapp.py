import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

file_path = 'data.csv'
data = pd.read_csv(file_path)

# Traitement des données (suppression des valeurs manquantes et des doublons)
data.dropna(axis=0, inplace=True)
data.drop_duplicates(inplace=True)

X = data.drop('charges', axis=1)
y = data['charges']

#Crée une colone de smoker en fonction du BMI
X['smoker_binary'] = (X['smoker'] == 'yes').astype(int)

#Création des intervalles pour les catégories BMI
bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]  # Les limites des catégories

#Étiquettes pour les catégories BMI
labels = [
    'underweight', 'normal weight', 'overweight',
    'obesity class I', 'obesity class II', 'obesity class III'
    ]

#Utilisation de pd.cut pour créer de nouvelles colonnes basées sur les catégories BMI
X['BMI_category'] = pd.cut(X['bmi'], bins=bins, labels=labels, right=False)

#Utilisation de pd.get_dummies pour obtenir des colonnes binaires pour chaque catégorie
BMI_dummies = pd.get_dummies(X['BMI_category'])

#Ajout des colonnes binaires au DataFrame X
X = pd.concat([X, BMI_dummies], axis=1)

X['bmi_smoker'] = X['bmi'] * X['smoker_binary']
X = X.drop('smoker_binary', axis=1)

#Suppression de la colonne 'BMI_category' car elle n'est plus nécessaire
X = X.drop('BMI_category', axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.85, random_state=42, stratify=X['smoker'])

# Création des pipelines pour le prétraitement des données
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

numerical_pipeline = Pipeline([
    ('poly', PolynomialFeatures(2)),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numerical_pipeline, numerical_cols),
        ('categorical', categorical_pipeline, categorical_cols)
    ])

# Création du pipeline complet
LR_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regression', LinearRegression())
])

# Entraînement du modèle
LR_pipeline.fit(X_train, y_train)

# Interface utilisateur Streamlit pour la prédiction
st.sidebar.header('Entrez les informations pour estimer les charges d\'assurance')

# Champs de saisie pour les caractéristiques
age = st.sidebar.number_input('Âge', min_value=0, max_value=100, value=30)
sex = st.sidebar.radio('Sexe', ['male', 'female'])
bmi = st.sidebar.number_input('Indice de masse corporelle (BMI)', min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.number_input('Nombre d\'enfants', min_value=0, max_value=10, value=0)
smoker = st.sidebar.radio('Fumeur', ['yes', 'no'])
region = st.sidebar.selectbox('Région', ['southwest', 'southeast', 'northwest', 'northeast'])

## # ... (ton code précédent jusqu'à la création de input_data)

# Création d'un nouveau DataFrame pour la prédiction
prediction_data = pd.DataFrame({'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children],
                                'smoker': [smoker], 'region': [region]})

# Conversion de 'sex' en variable numérique
prediction_data['sex'] = prediction_data['sex'].map({'male': 1, 'female': 0})

# Gestion des régions avec One-Hot Encoding
region_dummies = pd.get_dummies(prediction_data['region'], prefix='region')
prediction_data = pd.concat([prediction_data, region_dummies], axis=1)
prediction_data.drop(['region'], axis=1, inplace=True)

# Création de 'smoker_binary'
prediction_data['smoker_binary'] = (prediction_data['smoker'] == 'yes').astype(int)

# Création de 'BMI_category' et ses dummies
bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
labels = ['underweight', 'normal weight', 'overweight', 'obesity class I', 'obesity class II', 'obesity class III']
prediction_data['BMI_category'] = pd.cut(prediction_data['bmi'], bins=bins, labels=labels, right=False)
BMI_dummies = pd.get_dummies(prediction_data['BMI_category'])
prediction_data = pd.concat([prediction_data, BMI_dummies], axis=1)

# Calcul de 'bmi_smoker'
prediction_data['bmi_smoker'] = prediction_data['bmi'] * prediction_data['smoker_binary']

# Suppression des colonnes inutiles
prediction_data.drop(['BMI_category'], axis=1, inplace=True)

# Utilisation du même prétraitement que pour les données d'entraînement
input_preprocessed = preprocessor.transform(prediction_data)

# ... (le reste de ton code pour la prédiction)

# Utilisation du même prétraitement que pour les données d'entraînement
input_preprocessed = preprocessor.transform(prediction_data)

# Prédiction des charges d'assurance
predicted_charge = LR_pipeline.predict(input_preprocessed)

# Affichage de la prédiction
st.subheader('Estimation des charges d\'assurance')
st.write(f"Estimation des charges d\'assurance : {predicted_charge[0]}")

# Calcul du score R2 sur les données d'entraînement
y_pred_LR = LR_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred_LR)

st.subheader('Score R2 sur les données d\'entraînement')
st.write(f"Score R2 : {r2}")
