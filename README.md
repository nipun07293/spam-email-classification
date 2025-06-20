# spam-email-classification
Data Preprocessing

Load dataset

Rename and clean columns

Convert labels (spam/ham) to binary (1/0)

Split into train and test sets

Model Building

Use CountVectorizer for text vectorization

Apply 3 classifiers:

Multinomial Naive Bayes

Logistic Regression

Decision Tree

Model Evaluation

Custom evaluate_model() function computes:

ROC-AUC Scores

ROC Curves

Confusion Matrices

Classification Reports

Outputs precision, recall, F1-score, and accuracy for both train and test data

Spam Detection Function

detect_spam() function classifies new messages using any of the trained models.
