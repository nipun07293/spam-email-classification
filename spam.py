# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve, classification_report)
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Load dataset
df = pd.read_csv("abc.csv", encoding='ISO-8859-1')
df.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)
df.drop(columns={'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'}, inplace=True)
df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Spam, test_size=0.25, random_state=42)

# Evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    '''Evaluates classification model, plots ROC curve, confusion matrix, prints classification report'''
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    pred_prob_train = model.predict_proba(X_train)[:, 1]
    pred_prob_test = model.predict_proba(X_test)[:, 1]

    # ROC AUC Score
    roc_auc_train = roc_auc_score(y_train, pred_prob_train)
    roc_auc_test = roc_auc_score(y_test, pred_prob_test)
    print(f"\nTrain ROC AUC: {roc_auc_train:.4f}")
    print(f"Test ROC AUC: {roc_auc_test:.4f}")

    # ROC Curve
    fpr_train, tpr_train, _ = roc_curve(y_train, pred_prob_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, pred_prob_test)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_train, tpr_train, label=f"Train ROC AUC: {roc_auc_train:.2f}")
    plt.plot(fpr_test, tpr_test, label=f"Test ROC AUC: {roc_auc_test:.2f}")
    plt.legend()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    # Confusion Matrix
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.heatmap(cm_train, annot=True, xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                cmap="Oranges", fmt='g', ax=ax[0])
    ax[0].set_title("Train Confusion Matrix")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")
    sns.heatmap(cm_test, annot=True, xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                cmap="Oranges", fmt='g', ax=ax[1])
    ax[1].set_title("Test Confusion Matrix")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Classification Report
    print("\nTrain Classification Report:")
    print(classification_report(y_train, y_pred_train, digits=4))
    print("Test Classification Report:")
    print(classification_report(y_test, y_pred_test, digits=4))

    # Scores Summary
    precision_train = precision_score(y_train, y_pred_train)
    precision_test = precision_score(y_test, y_pred_test)
    recall_train = recall_score(y_train, y_pred_train)
    recall_test = recall_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)

    return [precision_train, precision_test, recall_train, recall_test,
            acc_train, acc_test, roc_auc_train, roc_auc_test, f1_train, f1_test]

# Multinomial Naive Bayes Model
nb_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
print("\n===== Multinomial Naive Bayes =====")
nb_scores = evaluate_model(nb_pipeline, X_train, X_test, y_train, y_test)

# Logistic Regression Model
lr_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('lr', LogisticRegression(max_iter=1000))
])
print("\n===== Logistic Regression =====")
lr_scores = evaluate_model(lr_pipeline, X_train, X_test, y_train, y_test)

# Decision Tree Classifier
dt_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('dt', DecisionTreeClassifier(random_state=42))
])
print("\n===== Decision Tree Classifier =====")
dt_scores = evaluate_model(dt_pipeline, X_train, X_test, y_train, y_test)

# Spam Detection Function using Logistic Regression (as example — you can swap with any)
def detect_spam(email_text, model):
    prediction = model.predict([email_text])
    return "This is a Spam Email!" if prediction[0] == 1 else "This is a Ham Email!"

# Test the spam detection function
sample_email = 'Free Tickets for IPL'
print("\n[Spam Check]")
print(detect_spam(sample_email, lr_pipeline))
