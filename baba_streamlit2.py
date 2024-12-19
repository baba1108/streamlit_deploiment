import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

st.markdown("""
    <style>
        /* Couleur de fond de la page */
        body {
            background-color: #f0f8ff;  /* Bleu clair */
        }

        /* Couleur du texte principal */
        .css-1d391kg {
            color: #2e8b57;  /* Vert forêt */
        }

        /* Changer la couleur des boutons */
        .css-1emrehy.edgvbvh3 {
            background-color: #ff6347;  /* Tomate */
            color: white;
        }

        /* Personnalisation des titres */
        h1 {
            color: #ff1493;  /* Rose profond */
        }

        h2 {
            color: #4682b4;  /* Bleu acier */
        }

        h3 {
            color: #8a2be2;  /* Bleu violet */
        }

    </style>
""", unsafe_allow_html=True)

st.title('streamlit checkpoint1')
st.header("Financial_inclusion_dataset.csv")

st.sidebar.title("Options")
show_data = st.sidebar.checkbox("Afficher les données", value=True)
show_histograms = st.sidebar.checkbox("Afficher les histogrammes", value=True)
show_model_evaluation = st.sidebar.checkbox("Afficher l'évaluation du modèle", value=True)

# Lire le fichier CSV dans un DataFrame pandas
data = pd.read_csv('Financial_inclusion_dataset.csv')
st.dataframe(data.head())
# Afficher les 5 premières lignes du DataFrame
st.write("Aperçu du Dataset :")
st.write("Informations du dataset :")
st.text(data.info())
st.write("Valeurs manquantes par colonne :")
st.write(data.isnull().sum())
print(data.head())
print(data.duplicated().sum())
print(data.describe())

# prompt: Sur la base de l'exploration des données précédentes, former et tester un classificateur d'apprentissage automatique

# Drop irrelevant columns
data = data.drop(['uniqueid', 'country'], axis=1)

# Convert categorical features to numerical using Label Encoding
label_encoders = {}
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Display the processed data
st.write("Données après encodage :")
st.dataframe(data.head())
print(data.info())

# Define features (X) and target (y)
X = data.drop('bank_account', axis=1)
y = data['bank_account']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increased max_iter
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
st.write(f"Précision du modèle : {accuracy}")

# Rapport de classification
st.write("Rapport de classification :")
st.text(classification_report(y_test, y_pred))




