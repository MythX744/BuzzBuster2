import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# Load preprocessed data from pickle
with open('pickle_file/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

preprocessed_tweets = data['preprocessed_tweets']
X = data['embeddings']
y = data['labels']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
# calculating metrics
precision_log = precision_score(y_test, y_pred)
recall_log = recall_score(y_test, y_pred)
f1_log = f1_score(y_test, y_pred)
accuracy_log = accuracy_score(y_test, y_pred)
conf_matrix_log = confusion_matrix(y_test, y_pred)

# print metrics

print(f'Precision: {precision_log * 100 :.2f} %')
print(f'Recall: {recall_log * 100 :.2f} %')
print(f'F1 Score: {f1_log* 100 :.2f} %')
print(f'Accuracy: {accuracy_log* 100 :.2f} %')
print(f'Confusion Matrix:\n{conf_matrix_log}')


# Save the model
with open('pickle_file/logistic_regression.pkl', 'wb') as f:
    pickle.dump(log_reg, f)

# Save the metrics
metrics = {
    'accuracy': accuracy_log,
    'f1': f1_log,
    'recall': recall_log,
    'precision': precision_log
}
pickle.dump(metrics, open('pickle_file/metrics_logistic_regression.pkl', 'wb'))
