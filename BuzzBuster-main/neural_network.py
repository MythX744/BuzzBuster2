import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Load preprocessed data from pickle
with open('pickle_file/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

preprocessed_tweets = data['preprocessed_tweets']
X = data['embeddings']
y = data['labels']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = np.array(y_train)
y_test = np.array(y_test)


# Ensure X has the correct shape for the neural network
X_train = np.array(X_train)
X_test = np.array(X_test)
if len(X_train.shape) == 2:
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]), dtype=tf.float32),
    tf.keras.layers.Flatten(),  # Flatten the output
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=3, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate metrics
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy_svm = accuracy_score(y_test, y_pred)
conf_matrix_svm = confusion_matrix(y_test, y_pred)

# Print metrics
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')
print(f'Accuracy: {accuracy_svm * 100:.2f}%')
print(f'Confusion Matrix:\n{conf_matrix_svm}')


# Save the model
with open('pickle_file/neural_network.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the metrics
metrics = {
    'accuracy': accuracy_svm,
    'f1': f1,
    'recall': recall,
    'precision': precision
}
with open('pickle_file/metrics_neural_network.pkl', 'wb') as f:
    pickle.dump(metrics, f)

