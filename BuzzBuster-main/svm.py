import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Load preprocessed data from pickle
with open('pickle_file/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

preprocessed_tweets = data['preprocessed_tweets']
X = data['embeddings']
y = data['labels']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear', C=10, gamma=0.1)
svm_model.fit(X_train, y_train)

# Predict using the SVM model
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# Print metrics
print(f'Precision: {precision_svm * 100 :.2f} %')
print(f'Recall: {recall_svm* 100 :.2f} %')
print(f'F1 Score: {f1_svm* 100 :.2f} %')
print(f'Accuracy: {accuracy_svm* 100 :.2f} %')
print(f'Confusion Matrix:\n{conf_matrix_svm}')

# Save the model
with open('pickle_file/svm.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Save the metrics
metrics = {
    'accuracy': accuracy_svm,
    'f1': f1_svm,
    'recall': recall_svm,
    'precision': precision_svm
}
with open('pickle_file/metrics_svm.pkl', 'wb') as f:
    pickle.dump(metrics, f)
