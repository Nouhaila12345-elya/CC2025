Analyse Prédictive appliquée au Dataset Bancaire
Pré-traitement, EDA et Modélisation Machine Learning

Nom de l'étudiante – ENCGS
Date : Aujourd’hui

Table des matières

Introduction

Méthodologie

Analyse Exploratoire des Données (EDA)

Modélisation

Conclusion

Figures

Introduction

Dans un contexte bancaire où la prise de décision doit être rapide, fiable et fondée sur des données, la prédiction du comportement des clients constitue un enjeu majeur.

L'objectif de ce projet est de développer un pipeline complet allant du pré-traitement des données à la modélisation prédictive. Le dataset bancaire analysé contient des informations socio-économiques, comportementales et transactionnelles.

Ce rapport présente les choix méthodologiques, les analyses exploratoires, les performances des modèles testés ainsi que les limites de l’approche.

Méthodologie
Pré-traitement des données
1. Nettoyage

Suppression des doublons et harmonisation des types de variables :

df.drop_duplicates(inplace=True)
df['Age'] = df['Age'].astype(int)
df['Balance'] = df['Balance'].astype(float)
df['Job'] = df['Job'].astype('category')

2. Imputation

Traitement des valeurs manquantes :

from sklearn.impute import KNNImputer

# Médiane pour variables numériques
df['Balance'].fillna(df['Balance'].median(), inplace=True)
# Catégorie "Unknown" pour variables catégorielles
df['Job'].fillna('Unknown', inplace=True)
# Imputation KNN pour variables complexes
imputer = KNNImputer(n_neighbors=5)
df[['Age','Balance','Duration']] = imputer.fit_transform(df[['Age','Balance','Duration']])

3. Encodage
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# One-Hot Encoding pour variables nominales
df = pd.get_dummies(df, columns=['Job', 'Marital'])
# Label Encoding pour variables ordinales
le = LabelEncoder()
df['Education'] = le.fit_transform(df['Education'])

4. Normalisation
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age','Balance','Duration']] = scaler.fit_transform(df[['Age','Balance','Duration']])

Analyse Exploratoire des Données (EDA)
Feature Engineering
# Création de tranches d'âges
df['AgeGroup'] = pd.cut(df['Age'], bins=[18,30,55,100], labels=['Jeune','Adulte','Senior'])
# Transformation logarithmique de la balance
df['Balance_log'] = np.log1p(df['Balance'])
# Indicateur si contacté auparavant
df['ContactedBefore'] = np.where((df['Campaign']>0) | (df['Previous']>0), 1, 0)

Modélisation
Algorithmes testés

Logistic Regression

Random Forest

XGBoost

Validation et optimisation
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
params = {'n_estimators':[100,200], 'max_depth':[5,10]}
grid = GridSearchCV(rf, params, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

Évaluation des modèles
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

y_pred = best_model.predict(X_test)
accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred)
roc_auc_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

Conclusion

Le modèle XGBoost offre les meilleures performances globales avec un ROC-AUC de 0.96.

Limites :

Dataset fortement déséquilibré

Certaines variables disponibles uniquement après contact

Risque de surapprentissage

Pistes d’amélioration :

Appliquer SMOTE pour rééquilibrage

Développer un modèle temps réel sans la variable duration

Tester des modèles de deep learning

Intégrer plus de variables socio-comportementales

Figures
Distribution de l’âge des clients

Boxplots des variables numériques

Matrice des corrélations

Courbes ROC des modèles

Matrice de confusion – modèle XGBoost
