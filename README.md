# Titre du Projet : Estimation de Primes d'Assurance avec IA
# Contexte du Projet
Lancée récemment, notre société de conseil en data et IA a été sélectionnée pour participer à un appel d'offres d'Assur’aimant, un assureur français s’implantant aux États-Unis.<br> Notre mission est de développer une solution IA permettant d'estimer les primes d'assurance pour le marché américain, en remplacement des méthodes traditionnelles longues et coûteuses basées sur des ratios et l'expérience des courtiers.

# Objectifs
Analyse Exploratoire des Données : Comprendre le profil des clients d'Assur’aimant à travers diverses analyses statistiques et visuelles.<br>
Modélisation Machine Learning : Développer un modèle prédictif pour estimer les primes d'assurance en fonction des données démographiques des clients.<br>

# Données
Le jeu de données comprend :<br>

Indice de masse corporelle (BMI)<br>
Sexe<br>
Âge<br>
Nombre d'enfants à charge<br>
Statut de fumeur<br>
Région résidentielle aux États-Unis<br>
Charges (prime d'assurance)<br>

# Fichiers du Projet
**app.py** : Application Streamlit pour la démonstration de la modélisation. <br>
**nettoyage.ipynb** : Nettoyage et préparation des données.<br>
**modelisation.ipynb** : Construction et évaluation des modèles de machine learning.<br>
**data.csv** : Jeu de données utilisé.<br>
**analyse.ipynb** : Analyse exploratoire des données.

# Installation
Suivez ces étapes pour configurer le projet :<br>
Clonez le dépôt.<br>
Exécutez app.py pour lancer l'application Streamlit.<br>

# Utilisation
**Analyse de Données** :
Lancez analyse.ipynb pour visualiser l'analyse exploratoire des données.
**Modélisation** :
Utilisez modelisation.ipynb pour voir les modèles de machine learning et leurs performances.
**Application Streamlit** : 
Lancez app.py pour utiliser l'application de prédiction des primes.<br>
lien de l'application en ligne : https://assurepeopleusa.streamlit.app/<br>
**Approche et Méthodologies**
Analyse univariée et bivariée, tests statistiques.<br>
Modélisation avec régression linéaire, Lasso, Ridge, et ElasticNet.<br>
Utilisation de pipelines, PolynomialFeatures, et sélection d'hyperparamètres.<br>
Interprétation des résultats et importance des variables.<br>


# Dépendances
installation des bibliothèques:<br>
Créer un environnement python:<br>
python3 -m venv venv<br>
source venv/bin/activate<br>
Requirement.text via la commande "pip -r requirements.txt"<br>
pour lancer streamlite il faut éxécute lance la commande "streamlit run app.py"<br>
lien de notre présentation : https://www.canva.com/design/DAF53kEs7GE/pWiLJYcQm7uhIBXb2MeKZQ/edit?utm_content=DAF53kEs7GE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton


