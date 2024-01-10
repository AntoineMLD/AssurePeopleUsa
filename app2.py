import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Fonction pour effectuer la prédiction
def predict_charges(age, sex, bmi, children, smoker, region):
    input_data = pd.DataFrame({'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children],
                               'smoker': [smoker], 'region': [region]})

    
    input_data['sex'] = 1 if sex == 'male' else 0
    input_data['smoker'] = 1 if smoker == 'yes' else 0
    input_data = pd.concat([input_data, pd.get_dummies(input_data['region'], prefix='is', prefix_sep='_')], axis=1)
    #input_data.drop('region', axis=1, inplace=True)

    bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
    labels = ['underweight', 'normal weight', 'overweight', 'obesity class I', 'obesity class II', 'obesity class III']
    input_data['BMI_category'] = pd.cut(input_data['bmi'], bins=bins, labels=labels, right=False)
    input_data = pd.concat([input_data, pd.get_dummies(input_data['BMI_category'])], axis=1)
    input_data['bmi_smoker'] = input_data['bmi'] * input_data['smoker']
    #input_data.drop(['BMI_category', 'smoker'], axis=1, inplace=True)

    return model.predict(input_data)[0]

# Chargement du modèle et du score R2
with open('exportModel.pkl', 'rb') as file:
    model = pickle.load(file)

with open('exportR2.pkl', 'rb') as file:
    r2 = pickle.load(file)

# Interface utilisateur Streamlit pour la saisie des données utilisateur
st.sidebar.header('Entrez vos informations pour comparaison et estimation des charges')
age = st.sidebar.number_input('Âge', min_value=0, max_value=100, value=30)
sex = st.sidebar.selectbox('Sexe', ['male', 'female'])
bmi = st.sidebar.number_input('Indice de masse corporelle (BMI)', min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.number_input('Nombre d\'enfants', min_value=0, max_value=5, value=0)
smoker = st.sidebar.selectbox('Fumeur', ['yes', 'no'])
smoker = 1 if smoker == 'yes' else 0
region = st.sidebar.selectbox('Région', ['southwest', 'southeast', 'northwest', 'northeast'])


# Appel de la fonction de prédiction
rounded_prediction = round(predict_charges(age, sex, bmi, children, smoker, region), 2)

# Affichage du score R2 et de l'estimation
st.write(f"Taux de prédibilité : {r2*100:.2f}%")
st.write(f"<span style='font-size:24px'>Estimation des charges d\'assurance :</span> <span style='color:red;font-size:24px'>{rounded_prediction:.2f}</span><span style='font-size:24px'> $</span>", unsafe_allow_html=True)

# Chargement du jeu de données et ajout des données utilisateur pour la visualisation
input_data = pd.DataFrame({'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children],
                               'smoker': [smoker], 'region': [region]})
file_path = 'data_cleaned.csv'
data = pd.read_csv(file_path)
data = pd.concat([data, pd.DataFrame(input_data.iloc[0]).transpose()], ignore_index=True)

# Création des graphiques
st.title("Comparaison de vos données avec l'ensemble des données")

# Histogramme de BMI avec la donnée de l'utilisateur
st.subheader("Histogramme du BMI")
fig, ax = plt.subplots()
sns.histplot(data['bmi'], kde=False, ax=ax)
ax.axvline(x=bmi, color='red', linestyle='--', label='Votre BMI')
ax.set_title("Distribution du BMI avec votre BMI")
ax.legend()
st.pyplot(fig)

# Histogramme de l'âge avec la donnée de l'utilisateur
st.subheader("Distribution de l'Âge")
fig, ax = plt.subplots()
sns.histplot(data['age'], bins=45, color='skyblue', ax=ax)
ax.axvline(x=age, color='red', linestyle='--', label='Votre Âge')
ax.set_title('Distribution de l\'Âge avec votre Âge')
ax.legend()
st.pyplot(fig)

# Distribution du Nombre d'Enfants avec la donnée de l'utilisateur
st.subheader("Distribution du Nombre d'Enfants")
fig, ax = plt.subplots()
sns.countplot(x='children', data=data, ax=ax, palette='viridis')
ax.axvline(x=children - 0.01, color='red', linestyle='--', label='Vos Enfants')
ax.set_title("Nombre d'Enfants avec votre Nombre d'Enfants")
ax.legend()
st.pyplot(fig)



# Distribution Fumeurs / Non-Fumeurs avec la donnée de l'utilisateur
st.subheader("Fumeurs vs Non-Fumeurs")
fig, ax = plt.subplots()
sns.countplot(x='smoker', data=data, ax=ax, palette='coolwarm')
ax.axvline(x=smoker - 0.01, color='red', linestyle='--', label='Votre Statut de Fumeur')
ax.set_title("Fumeurs vs Non-Fumeurs avec votre Statut de Fumeur")
ax.legend()
st.pyplot(fig)