import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, LabelEncoder,  FunctionTransformer
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def preprocess_data(X):
    # Initialiser l'encodeur de labels pour la conversion des données catégorielles
    label_encoder = LabelEncoder()

    # Convertir les colonnes 'sex' et 'smoker' en valeurs entières
    columns_to_encode = ['sex', 'smoker']
    for column in columns_to_encode:
        X[column] = label_encoder.fit_transform(X[column])  # Encodage des colonnes

    # Créer des variables binaires avec des noms explicites pour la colonne 'region'
    region_dummies_named = pd.get_dummies(X['region'], prefix='is', prefix_sep='_')

    # Concaténer ces variables binaires avec le DataFrame original
    X_with_named_dummies = pd.concat([X, region_dummies_named], axis=1)

    # Création d'une colonne binaire pour le statut de fumeur
    X_with_named_dummies['smoker_binary'] = (X_with_named_dummies['smoker'] == 1).astype(int)

    # Définition des intervalles pour les catégories de BMI
    bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]  # Limites des catégories
    labels = ['underweight', 'normal weight', 'overweight', 'obesity class I', 'obesity class II', 'obesity class III']

    # Utilisation de pd.cut pour créer de nouvelles colonnes basées sur les catégories de BMI
    X_with_named_dummies['BMI_category'] = pd.cut(X_with_named_dummies['bmi'], bins=bins, labels=labels, right=False)

    # Utilisation de pd.get_dummies pour obtenir des colonnes binaires pour chaque catégorie de BMI
    BMI_dummies = pd.get_dummies(X_with_named_dummies['BMI_category'])

    # Ajout de ces colonnes binaires au DataFrame X
    X_processed = pd.concat([X_with_named_dummies, BMI_dummies], axis=1)

    # Suppression des colonnes temporaires si nécessaire
    # X_processed = X_processed.drop(['smoker_binary', 'BMI_category'], axis=1)
    num_cols = len(X_processed.columns)
    # print(f"Nombre de colonnes : {num_cols}")
    return X_processed