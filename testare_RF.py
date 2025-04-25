import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

# --- Incarcare baza de date  ---
external_test = pd.read_csv("C:/Downloads/TREC_05_mic.csv")
external_test = external_test[['sender', 'subject', 'body', 'label']]

#preprocesarea e-mailurilor din baza de date de test
def preprocess_text(text):
    if not isinstance(text, str):  # If text is not a string (e.g., NaN or float), convert it
        text = str(text) if text is not None else ""
    text=text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

#concatenarea coloanelor de interes intr-o singura coloana si aplicarea functiei de procesare
external_test['text_cleaned'] = external_test['sender'] + " " + external_test['subject'] + " " + external_test['body']
external_test['text_cleaned'] = external_test['text_cleaned'].apply(preprocess_text)

# --- Load vectorizer and transform text ---
with open("vectorizer2.pkl", "rb") as f:
    loaded_vectorizer = pickle.load(f)

X_ext = external_test['text_cleaned']
y_ext = external_test['label']

X_final_vect = loaded_vectorizer.transform(X_ext).toarray()  # Convert to NumPy array

with open("rf_model_best.pkl", 'rb') as f:
    clasificator_random_forest = pickle.load(f)

# --- Predict and evaluate on external test set ---
y_ext_pred = clasificator_random_forest.predict(X_final_vect)
print("\nAcuratete pe baza de date de test:", accuracy_score(y_ext, y_ext_pred))
print("\nRaport de clasificare:\n", classification_report(y_ext, y_ext_pred))

# --- Confusion matrix for external test set ---
conf_matrix_ext = confusion_matrix(y_ext, y_ext_pred)
sns.heatmap(conf_matrix_ext, annot=True, fmt='d', cmap='Greens')
plt.title("Matricea de confuzie - baza de date de test")
plt.xlabel("Prezis")
plt.ylabel("Actual")
plt.show()

