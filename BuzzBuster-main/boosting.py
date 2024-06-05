import pickle
import xgboost as xgb
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

# Train a XGBoost model
model = xgb.XGBClassifier(
 learning_rate =0.01,
 n_estimators=1500,
 max_depth=5,
 min_child_weight=1,
 gamma=0.2,
 subsample=0.9,
 colsample_bytree=0.6,
 reg_alpha=65,
 objective= 'binary:logistic',
 nthread=15,
 scale_pos_weight=18,
 seed=27)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

conf_matrix_log = confusion_matrix(y_test, y_pred)

# Save the model
with open('pickle_file/xgboost.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the metrics
metrics = {
    'accuracy': accuracy,
    'f1': f1_score,
    'recall': recall,
    'precision': precision
}

print(f'Precision: {precision* 100 :.2f} %')
print(f'Recall: {recall* 100 :.2f} %')
print(f'F1 Score: {f1_score* 100 :.2f} %')
print(f'Accuracy: {accuracy* 100 :.2f} %')
print(f'Confusion Matrix:\n{conf_matrix_log}')

pickle.dump(metrics, open('pickle_file/metrics_xgboost.pkl', 'wb'))
