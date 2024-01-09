# AssurePeopleUsa
Projet de la formation Simplon: Prédire une prime d’assurance grâce à l’IA

# Titre du Projet : Estimation de Primes d'Assurance avec IA
# Contexte du Projet
Lancée récemment, notre société de conseil en data et IA a été sélectionnée pour participer à un appel d'offres d'Assur’aimant, un assureur français s’implantant aux États-Unis. Notre mission est de développer une solution IA permettant d'estimer les primes d'assurance pour le marché américain, en remplacement des méthodes traditionnelles longues et coûteuses basées sur des ratios et l'expérience des courtiers.

# Objectifs
Analyse Exploratoire des Données : Comprendre le profil des clients d'Assur’aimant à travers diverses analyses statistiques et visuelles.
Modélisation Machine Learning : Développer un modèle prédictif pour estimer les primes d'assurance en fonction des données démographiques des clients.

# Données
Le jeu de données comprend :

Indice de masse corporelle (BMI)
Sexe
Âge
Nombre d'enfants à charge
Statut de fumeur
Région résidentielle aux États-Unis
Charges (prime d'assurance)

# Fichiers du Projet
**app.py** : Application Streamlit pour la démonstration de la modélisation.
**nettoyage.ipynb** : Nettoyage et préparation des données.
**modelisation.ipynb** : Construction et évaluation des modèles de machine learning.
**data.csv** : Jeu de données utilisé.
**analyse.ipynb** : Analyse exploratoire des données.

# Installation
Suivez ces étapes pour configurer le projet :
Clonez le dépôt.
Exécutez app.py pour lancer l'application Streamlit.

# Utilisation
**Analyse de Données** :
Lancez analyse.ipynb pour visualiser l'analyse exploratoire des données.
**Modélisation** :
Utilisez modelisation.ipynb pour voir les modèles de machine learning et leurs performances.
**Application Streamlit** : 
Lancez app.py pour utiliser l'application de prédiction des primes.
**Approche et Méthodologies**
Analyse univariée et bivariée, tests statistiques.
Modélisation avec régression linéaire, Lasso, Ridge, et ElasticNet.
Utilisation de pipelines, PolynomialFeatures, et sélection d'hyperparamètres.
Interprétation des résultats et importance des variables.


# Dépendances
installation des bibliothèques:
Créer un environnement python:
python3 -m venv venv
source venv/bin/activate
Requirement.text via la commande "pip -r requirements.txt"
pour lancer streamlite il faut éxécute lance la commande "streamlit run app.py"


