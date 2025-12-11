# **Analyse Prédictive appliquée au Dataset Bancaire**

 ## Pré-traitement, EDA et Modélisation Machine Learning

El yazidi Nouhaila – ENCGS
Date : 11/12/2025

<img width="500" height="600" alt="image" src="https://github.com/user-attachments/assets/582e6298-542b-43b7-b21f-7d07673c31df" />

- Table des matières

- Introduction

- Méthodologie

- Analyse Exploratoire des Données (EDA)

- Modélisation

- Conclusion

### Introduction

Dans un contexte bancaire où la prise de décision doit être rapide, fiable et fondée sur des données, la prédiction du comportement des clients constitue un enjeu majeur.

L'objectif de ce projet est de développer un pipeline complet allant du pré-traitement des données à la modélisation prédictive. Le dataset bancaire à analyser contient des informations socio-économiques, comportementales et transactionnelles des clients.

Ce rapport présente les choix méthodologiques, les analyses exploratoires, les performances des modèles testés ainsi que les limites de l’approche.

### Méthodologie
Pré-traitement des données
Nettoyage

Les doublons ont été supprimés afin d’assurer une base fiable. Les types ont été harmonisés (variables transformées en int, float, category).

Imputation

Les valeurs manquantes ont été traitées par :

la médiane pour les variables numériques (robuste aux valeurs extrêmes)

la catégorie "Unknown" pour certaines variables catégorielles

une imputation KNN pour les variables ayant une structure complexe

### Encodage

Selon la nature des variables :

One-Hot Encoding pour les variables nominales

Label Encoding pour les variables ordinales

Target Encoding pour les variables catégorielles fortement corrélées à la cible

### Normalisation

Une standardisation par Z-score a permis d’homogénéiser les échelles des variables avant la modélisation, particulièrement utile pour SVM et KNN.

Analyse Exploratoire des Données (EDA)
Visualisation des distributions
<img width="994" height="739" alt="image" src="https://github.com/user-attachments/assets/75d544c1-8d5b-478a-8f49-72f9caf4daac" />

La figure ci-dessous montre la distribution des âges, révélant une concentration entre 25 et 55 ans.

### Détection d'outliers

Les boxplots indiquent la présence de valeurs extrêmes notamment dans les variables balance et duration.
<img width="1003" height="528" alt="image" src="https://github.com/user-attachments/assets/8477340b-e89e-4d0a-93a8-deedaba00706" />

### Analyse des corrélations
<img width="900" height="759" alt="image" src="https://github.com/user-attachments/assets/e2ebca4c-5ccd-4b60-9cca-d90a6232b889" />

La heatmap met en évidence une corrélation positive importante entre duration et la variable cible. Cette variable joue un rôle essentiel dans la prédiction.

Feature Engineering

Pour améliorer la pertinence du dataset :

Création d’une variable AgeGroup (jeune, adulte, senior)

Transformation logarithmique de balance

Création d’un indicateur ContactedBefore basé sur campaign et previous

### Modélisation
Algorithmes testés

Trois modèles ont été sélectionnés :

Logistic Regression

Random Forest

XGBoost

Validation et optimisation

Une Cross-Validation à 5 folds a été appliquée. L’optimisation des hyperparamètres a été réalisée via GridSearchCV.

Performances des modèles
Modèle	Accuracy	F1-score	ROC-AUC	RMSE
Logistic Regression	0.84	0.79	0.89	0.41
Random Forest	0.91	0.88	0.94	0.33
XGBoost	0.93	0.90	0.96	0.29
Courbe ROC

### Analyse des erreurs
<img width="519" height="435" alt="image" src="https://github.com/user-attachments/assets/2dc36081-0943-4d7f-b1a3-6e45b49905e9" />

La matrice de confusion de XGBoost montre que :

Les faux positifs restent modérés

Les faux négatifs sont faibles, ce qui est important pour un modèle bancaire

### Conclusion

Le modèle XGBoost offre les meilleures performances globales avec un ROC-AUC de 0.96.

### Limites :

Le dataset est fortement déséquilibré

Certaines variables (ex : duration) sont connues uniquement post-contact

Un risque de surapprentissage existe malgré la régularisation

Pistes d’amélioration :

Appliquer du SMOTE pour le rééquilibrage

Développer un modèle temps-réel sans duration

Tester des modèles de deep learning

Intégrer plus de variables socio-comportementales
