Phishing Attack Detection Using Machine Learning algorithms

This project focuses on detecting phishing attacks through machine learning techniques. By leveraging Support Vector Machines (SVM), Random Forest, and Logistic Regression, the project aims to classify emails as either phishing or legitimate with high accuracy.

Overview
Phishing attacks remain a major cybersecurity threat. This project utilizes machine learning models trained on a large dataset to identify phishing emails and distinguish them from legitimate ones. The objective is to contribute to a more secure online environment by automating phishing detection.

Dataset
We concatenated four publicly available datasets from Kaggle (CEAS, Nazario, Nigerian Fraud, and SpamAssassin Datasets from https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) containing phishing and legitimate emails to create a unified, larger dataset. This combined dataset was preprocessed and used to train the machine learning models.

For testing, we used the TREC_05_mic dataset, which is already included in this repository.

Preprocessing
The text in the datasets was preprocessed and transformed into numerical values to make it suitable for machine learning algorithms. The preprocessing steps included:

Text cleaning
Feature extraction (e.g., TF-IDF)
All preprocessing steps were applied uniformly to ensure that all models received the same input data.

Models
Three machine learning models were trained and evaluated:

Support Vector Machines (SVM)
Random Forest
Logistic Regression
The trained models were saved using Python's pickle library and can be loaded for inference or further testing.

Workflow
Preprocessing: The text data was preprocessed uniformly for all models.
Training: The three models were trained on the unified dataset.
Testing: The trained models were tested on the TREC_05_mic dataset to evaluate their performance.

Files in this Repository
We provide seven scripts in the repository: two scripts (one for training and one for testing) per model and an additional script used to obtain and preprocess the training dataset.
