import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def cleaner(table):
    #Crée une colone de smoker en fonction du BMI
    table['smoker_binary'] = (table['smoker'] == 'yes').astype(int)

    #Création des intervalles pour les catégories BMI
    bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]  # Les limites des catégories

    #Étiquettes pour les catégories BMI
    labels = [
        'underweight', 'normal weight', 'overweight',
        'obesity class I', 'obesity class II', 'obesity class III'
        ]

    #Utilisation de pd.cut pour créer de nouvelles colonnes basées sur les catégories BMI
    table['BMI_category'] = pd.cut(table['bmi'], bins=bins, labels=labels, right=False)

    #Utilisation de pd.get_dummies pour obtenir des colonnes binaires pour chaque catégorie
    BMI_dummies = pd.get_dummies(table['BMI_category'])

    #Ajout des colonnes binaires au DataFrame X
    table = pd.concat([table, BMI_dummies], axis=1)

    table['bmi_smoker'] = table['bmi'] * table['smoker_binary']
    table = table.drop('smoker_binary', axis=1)

    #Suppression de la colonne 'BMI_category' car elle n'est plus nécessaire
    table = table.drop('BMI_category', axis=1)
    return(table)


file_path = 'data.csv'

data = pd.read_csv(file_path)


# Vérification des informations manquantes et des doublons
missing_data = data.isnull().sum()
duplicates = data.duplicated().sum()
data = data.drop_duplicates()
data.dropna(axis=0, inplace=True)



X = data.drop('charges', axis=1)
y = data['charges']

X = cleaner(X)

#Affichage du DataFrame avec les nouvelles colonnes binaires pour les catégories BMI
# 80% pour train et 20% de test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.85, random_state=42, stratify=X['smoker'])



# Identifier les colonnes catégories et numériques
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Créer le pipeline pour les features numériques
numerical_pipeline = Pipeline([
    ('poly', PolynomialFeatures(2)),
    ('scaler', StandardScaler()) # Ajout de PolynomialFeatures
])


# Créer le pipeline pour les features catégorielles
categorial_pipeline = Pipeline([
    ('encoder', OneHotEncoder()),
    ('poly', PolynomialFeatures(2))
])


# Combine les pipelines en utilisant ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numerical_pipeline, numerical_cols),
        ('categorial', categorial_pipeline, categorical_cols)
    ])


LR_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regression', LinearRegression())
])





# On entraine les donnnées
LR_pipeline.fit(X_train, y_train)

# On predicte Linear Regression
y_pred_LR = LR_pipeline.predict(X_test)

# Calcul du score R2 sur les données d'entraînement
y_pred_LR = LR_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred_LR)

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

input_data = cleaner(input_data)


# X_test.columns
# input_data.columns

y_pred_input = LR_pipeline.predict(input_data)
# Affichage de la prédiction
st.subheader('Estimation des charges d\'assurance')
st.write(f"Estimation des charges d\'assurance : {y_pred_input[0]}")

# Affichage du score R2 sur les données d'entraînement
st.subheader('Score R2 sur les données d\'entraînement')
st.write(f"Score R2 : {r2}")
