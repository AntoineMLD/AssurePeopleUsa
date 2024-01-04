import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données depuis le fichier CSV
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Remplacer les valeurs catégoriques par des valeurs numériques
data["sex"].replace(['male', 'female'], [0, 1], inplace=True)
data["smoker"].replace(['no', 'yes'], [0, 1], inplace=True)

# Créer une copie des données sans la colonne 'region' pour la modélisation
data_model = data.drop("region", axis=1)

# Supprimer les lignes avec des valeurs manquantes
data_model.dropna(axis=0, inplace=True)

# Diviser les données en features (X) et la cible (y)
X = data_model.drop('charges', axis=1)
y = data_model['charges']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle de régression linéaire
model = LinearRegression()

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.title('Analyse des données')

# Afficher les métriques d'évaluation
st.write(f"Coût (MSE): {mse}")
st.write(f"R-squared: {r2}")

# Ajouter la colonne 'region' pour colorer les points dans les graphiques
data['color'] = data['region'].map({'northeast': 'green', 'southeast': 'blue', 'southwest': 'red', 'northwest': 'orange'})

# Obtenir les noms des colonnes de X
feature_names = X.columns

# Créer des nuages de points pour chaque colonne de X par rapport à y avec couleurs par région
for feature in feature_names:
    fig, ax = plt.subplots(figsize=(6, 4))
    for region, color in data.groupby('region')['color']:
        ax.scatter(data[data['region'] == region][feature], data[data['region'] == region]['charges'], label=region, color=color, alpha=0.5)
    ax.set_title(f'{feature} vs Charges')
    ax.set_xlabel(feature)
    ax.set_ylabel('Charges')
    ax.legend()
    st.pyplot(fig)
