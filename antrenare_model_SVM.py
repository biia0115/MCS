# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 12:06:59 2025

@author: bianc
"""
#incarcarea librariilor necesare

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# descarcarea unei baze de date a cuvintelor de legatura din limba engleza
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# incarcarea setului de date dupa prelucrare (concatenarea celor 4 baze de date gasite pe Kaggle + eliminarea randurilor unde se gasesc valori de tip NaN + pastrarea doar a coloanelor body subject sender si label)
phishing_db = pd.read_csv(r"C:\Users\bianc\Desktop\Proiect MCS\baza_de_date_finala.csv")


# definirea unei functii pentru prelucrarea sirurilor din baza de date (eliminare caractere speciale, scriere cu litera mica, eliminare cuvinte de legatura si numere)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# concatenarea stringurilor din coloanele sender body si subject intr-o singura coloana
phishing_db['text_cleaned'] = phishing_db['sender'] + " " + phishing_db['subject'] + " " + phishing_db['body']
#aplicarea preprocesarii coloanei text_cleaned
phishing_db['text_cleaned'] = phishing_db['text_cleaned'].apply(preprocess_text)

# definirea vectorului de caracteristici (X - coloana text_cleaned procesata) si a vectorului de etichete y (coloana label)
X = phishing_db['text_cleaned']
y = phishing_db['label']

# convertirea textului in valori numerice
vectorizer = TfidfVectorizer(stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2)
X_tfidf = vectorizer.fit_transform(X).toarray()  # Convert to NumPy array

#salvare vectorizer pentru a fi aplicat mai tarziu pe valorile de test
import pickle
with open(r'C:\Users\bianc\Desktop\Proiect MCS\vectorizer_ana.pkl','wb') as f:
    pickle.dump(vectorizer,f)



# impartirea datelor in date de antrenament si date de test (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)


#best SVM din gridsearch avea C=10, scale si rbf
#initializarea modelului SVM

svm_model = SVC(kernel='rbf',C=1,gamma='scale')

#antrenarea modelului
svm_model.fit(X_train, y_train)

#realizarea predictiei pe datele de test
y_pred = svm_model.predict(X_test)

# Evaluarea modelului 
acuratete = accuracy_score(y_test, y_pred)
matrice_confuzie = confusion_matrix(y_test, y_pred)

print(f"Acuratete antrenare: {acuratete:.4f}")
print("Matricea de confuzie:\n", matrice_confuzie)


#salvarea modelului pentru a-l putea aplica pe date noi
import pickle
with open(r'C:\Users\bianc\Desktop\Proiect MCS\model_SVM_C=1_vectorizer_ana.pkl','wb') as f:
    pickle.dump(svm_model,f)

