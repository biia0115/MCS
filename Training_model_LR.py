# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 12:45:17 2025

@author: Ana Brostic
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# --- Download stopwords ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Load and select relevant columns ---
df = pd.read_csv("D:/Master 1/Proiect/phishing_db1.csv")
df = df[['sender', 'subject', 'body', 'label']]

# --- Clean missing and empty text entries ---
df = df.dropna(subset=["sender", "subject", "body", "label"])
df = df[
    (df["sender"].str.strip() != "") &
    (df["subject"].str.strip() != "") &
    (df["body"].str.strip() != "")
].reset_index(drop=True)

# --- Save cleaned dataset ---
df.to_csv("D:/Master 1/Proiect/phishing_db1_clean.csv", index=False)
print(f" Cleaned dataset saved. Total rows: {len(df)}")

# --- Text preprocessing function ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

# --- Applying preprocessing to combined text ---
df['text_cleaned'] = (df['sender'] + " " + df['subject'] + " " + df['body']).apply(preprocess_text)

# --- Features and labels ---
X = df['text_cleaned']
y = df['label']

# --- TF-IDF vectorization with enhancements ---
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)
X_tfidf = vectorizer.fit_transform(X)

# --- Save vectorizer ---
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# --- Train/test split ---
X_train, X_test, Y_train, Y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Logistic Regression model ---
model = LogisticRegression(
    C=0.3,
    class_weight="balanced",
    max_iter=1000
)
model.fit(X_train, Y_train)

# --- Predict and evaluate ---
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)

# --- Print evaluation ---
print(f"Test accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# --- Save model ---
with open('model_final', 'wb') as f:
    pickle.dump(model, f)
print(" Model saved as model_final")
