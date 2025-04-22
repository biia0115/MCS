# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 12:28:45 2025

@author: bianc
"""
#importarea librariilor necesare
import pickle
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

#incrcarea bazei de date de test
phishing_db=pd.read_csv(r"C:\Users\bianc\Desktop\Proiect MCS\TREC_05_mic.csv")

#pastrarea coloanelor de interes
phishing_db = phishing_db[['sender', 'subject', 'body', 'label']]


#preprocesarea e-mailurilor din baza de date de test
def preprocesare_text(text):
    if not isinstance(text, str):  # If text is not a string (e.g., NaN or float), convert it
        text = str(text) if text is not None else ""
    text=text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text
#concatenarea coloanelor de interes intr-o singura coloana si aplicarea functiei de procesare
phishing_db['text_cleaned'] = phishing_db['sender'] + " " + phishing_db['subject'] + " " + phishing_db['body']
phishing_db['text_cleaned'] = phishing_db['text_cleaned'].apply(preprocesare_text)



#impartire in caracteristici si etichete
X=phishing_db['text_cleaned']
y=phishing_db['label']


#Transformare text in valori numerice
with open(r"C:\Users\bianc\Desktop\Proiect MCS\vectorizer_ana.pkl", 'rb') as f:
    vectorizer = pickle.load(f)

X_final_vect = vectorizer.transform(X).toarray()  # Convert to NumPy array


#incarcare model SVM antrenat pentru realizarea predictiilor pe date noi
with open(r"C:\Users\bianc\Desktop\Proiect MCS\model_SVM_C=1_vectorizer_ana.pkl", 'rb') as f:
    clasificator_SVM = pickle.load(f)

y_prezis=clasificator_SVM.predict(X_final_vect)
#evaluarea pe date noi

acuratete_test=accuracy_score(y,y_prezis)
matrice_confuzie_test=confusion_matrix(y, y_prezis)

print(f"Acuratete: {acuratete_test:.4f}")
print("Matricea de confuzie:\n", matrice_confuzie_test)