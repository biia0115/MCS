#cod cu gridsearch pentru antrenare

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---  Stopwords ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Încărcare baza de date ---
phishing_db = pd.read_csv("C:/Downloads/baza_de_date_finala.csv")

# --- Preprocesare text ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

# --- Curățare text + concatenare coloane ---
phishing_db['text_cleaned'] = phishing_db['sender'] + " " + phishing_db['subject'] + " " + phishing_db['body']
phishing_db['text_cleaned'] = phishing_db['text_cleaned'].apply(preprocess_text)

# --- Vectorizare TF-IDF ---
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2), min_df=2)
X = vectorizer.fit_transform(phishing_db['text_cleaned']).toarray()
y = phishing_db['label']

# --- Salvare vectorizer ---
with open("vectorizer2.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# --- Împărțire set de date ---
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Definire model și hiperparametri pentru RandomizedSearchCV ---
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf = RandomForestClassifier(random_state=42)

# --- RandomizedSearchCV ---
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='accuracy'
)

random_search.fit(X_train, Y_train)
best_model = random_search.best_estimator_

# --- Salvare model antrenat optimizat ---
with open("rf_model_best.pkl", "wb") as f:
    pickle.dump(best_model, f)


# --- Evaluare pe setul de testare ---
y_pred = best_model.predict(X_test)
print("Acuratete pe baza de date de antrenare:", accuracy_score(Y_test, y_pred))
print("\nRaport de clasificare:\n", classification_report(Y_test, y_pred))

# --- Matrice de confuzie ---
conf_matrix = confusion_matrix(Y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Matricea de confuzie - baza de date de antrenare")
plt.xlabel("Prezis")
plt.ylabel("Actual")
plt.show()

