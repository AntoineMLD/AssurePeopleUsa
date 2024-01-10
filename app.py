import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
file_path = 'data.csv'

data = pd.read_csv(file_path)


# Vérification des informations manquantes et des doublons
missing_data = data.isnull().sum()
duplicates = data.duplicated().sum()
data = data.drop_duplicates()

# Afficher le DataFrame avec les nouvelles colonnes binaires
print(data.head())

data.dropna(axis=0, inplace=True)


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

#Affichage du DataFrame avec les nouvelles colonnes binaires pour les catégories BMI
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.85, random_state=42, stratify=X['smoker'])
# 80% pour train et 20% de test

print("Train set X", X_train.shape)
print("Train set Y", y_train.shape)
print("Test set X", X_test.shape)
print("Test set Y", y_test.shape)


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




# Créer le pipeline final en ajoutant le model

LR_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regression', LinearRegression())
])

# Lasso_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('Lasso', Lasso())
# ])

# ElasticNet_pipeline = Pipeline([
#     ('prepocessor', preprocessor),
#     ('ElasticNet', ElasticNet())
# ])

print (len(X_train))
print (len(y_train))

# On entraine les donnnées
LR_pipeline.fit(X_train, y_train)
# Lasso_pipeline.fit(X_train, y_train)
# ElasticNet_pipeline.fit(X_train, y_train)

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
children = st.sidebar.number_input('Nombre d\'enfants', min_value=0, max_value=10, value=0)
smoker = st.sidebar.radio('Fumeur', ['yes', 'no'])
region = st.sidebar.selectbox('Région', ['southwest', 'southeast', 'northwest', 'northeast'])

# Préparation des données pour la prédiction
input_data = pd.DataFrame({'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children],
                           'smoker': [smoker], 'region': [region]})

# Création de la colonne binaire pour le fumeur
input_data['smoker_binary'] = (input_data['smoker'] == 'yes').astype(int)
input_data.drop('smoker', axis=1, inplace=True)

# Création des catégories BMI et de leurs colonnes binaires correspondantes
bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
labels = ['underweight', 'normal weight', 'overweight', 'obesity class I', 'obesity class II', 'obesity class III']
input_data['BMI_category'] = pd.cut(input_data['bmi'], bins=bins, labels=labels, right=False)
BMI_dummies = pd.get_dummies(input_data['BMI_category'])
input_data = pd.concat([input_data, BMI_dummies], axis=1)
input_data['bmi_smoker'] = input_data['bmi'] * input_data['smoker_binary']
input_data.drop(['bmi', 'BMI_category'], axis=1, inplace=True)




# Préparation des données pour la prédiction
input_data = pd.DataFrame({'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children],
                           'smoker': [smoker], 'region': [region]})

# Prédiction des charges d'assurance
predicted_charge = LR_pipeline.predict(input_data)

# Affichage de la prédiction
st.subheader('Estimation des charges d\'assurance')
st.write(f"Estimation des charges d\'assurance : {predicted_charge[0]}")

# Affichage du score R2 sur les données d'entraînement
st.subheader('Score R2 sur les données d\'entraînement')
st.write(f"Score R2 : {r2}")
