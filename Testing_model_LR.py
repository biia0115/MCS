# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 12:49:36 2025

@author: Ana Brostic
"""

import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Download stopwords ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Load test dataset ---
df = pd.read_csv("D:/Master 1/Proiect/TREC_05_mic.csv") 

# --- Select and clean only required columns ---
df = df[['sender', 'subject', 'body', 'label']]  

# --- Drop rows with NaN or empty text ---
df = df.dropna(subset=["sender", "subject", "body", "label"])
df = df[
    (df["sender"].str.strip() != "") &
    (df["subject"].str.strip() != "") &
    (df["body"].str.strip() != "")
].reset_index(drop=True)

# --- Preprocessing function ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

# --- Apply preprocessing to combined fields ---
df['text_cleaned'] = (
    df['sender'] + " " + df['subject'] + " " + df['body']
).apply(preprocess_text)

# --- Load vectorizer and transform data ---
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

X_test = vectorizer.transform(df['text_cleaned'])

# --- Load trained model ---
with open("model_final", "rb") as f:
    model = pickle.load(f)

# --- Predict and evaluate ---
y_true = df['label']
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print(f" Test accuracy: {accuracy:.4f}")
print(" Confusion Matrix:")
print(conf_matrix)
