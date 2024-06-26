import string
import joblib
import pickle
import numpy as np
import torch
import re
import pandas as pd
from torch import nn
from transformers import BertModel
from emoji.core import demojize
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits


# Load the trained models
log_reg = joblib.load('pickle_file/logistic_regression.pkl')
svm_model = joblib.load('pickle_file/svm.pkl')
neural_network = joblib.load('pickle_file/neural_network.pkl')
xgboost = joblib.load('pickle_file/xgboost.pkl')

# Load the metrics
log_reg_metrics = joblib.load('pickle_file/metrics_logistic_regression.pkl')
svm_metrics = joblib.load('pickle_file/metrics_svm.pkl')
xgboost_metrics = joblib.load('pickle_file/metrics_xgboost.pkl')
neural_network_metrics = joblib.load('pickle_file/metrics_neural_network.pkl')

# Load the BERT model and metrics
model_path = 'pickle_file/bert_classifier_cpu.pkl'
with open(model_path, 'rb') as f:
    bert_classifier = pickle.load(f)

metrics_path = 'pickle_file/metrics_bert_classifier_cpu.pkl'
with open(metrics_path, 'rb') as f:
    bert_metrics = pickle.load(f)

# Load BERTweet tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModel.from_pretrained("vinai/bertweet-base")

# Load preprocessed data from pickle
with open('pickle_file/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

preprocessed_tweets = data['preprocessed_tweets']
X = data['embeddings']
y = data['labels']

print("Preprocessed tweets and embeddings loaded from 'preprocessed_data.pkl'.")


# Function to preprocess tweets
def preprocess_tweet(tweet, max_length=128):
    if isinstance(tweet, float):
        return ""

    # Lowercasing
    tweet = tweet.lower()
    # Removing URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    # Removing mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Removing hashtags (keeping the word)
    tweet = re.sub(r'#', '', tweet)
    # Removing punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Replace special characters
    tweet = re.sub(r"[^\w\s]", " ", tweet)
    # Remove extra whitespaces
    tweet = re.sub(r"\s+", " ", tweet).strip()
    # Truncate long tweets
    if len(tweet) > max_length:
        tweet = tweet[:max_length]
    return tweet


# Function to get BERTweet embeddings for a single tweet
def get_bertweet_embeddings(tweet, tokenizer, model, max_length=128):
    try:
        print(f"Processing tweet: {tweet}")
        inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        print("Tokenization successful.")
        with torch.no_grad():
            outputs = model(**inputs)
        print("Embedding extraction successful.")
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    except Exception as e:
        print(f"Error processing tweet: {tweet}")
        print(f"Exception: {e}")
        return np.zeros(model.config.hidden_size)


# Functions for metrics
def logistic_regression_metrics():
    accuracy_log = log_reg_metrics['accuracy'] * 100
    f1_log = log_reg_metrics['f1'] * 100
    recall_log = log_reg_metrics['recall'] * 100
    precision_log = log_reg_metrics['precision'] * 100
    return f"{accuracy_log:.2f}%", f"{f1_log:.2f}%", f"{recall_log:.2f}%", f"{precision_log:.2f}%"


def metrics_svm():
    accuracy_svm = svm_metrics['accuracy'] * 100
    f1_svm = svm_metrics['f1'] * 100
    recall_svm = svm_metrics['recall'] * 100
    precision_svm = svm_metrics['precision'] * 100
    return f"{accuracy_svm:.2f}%", f"{f1_svm:.2f}%", f"{recall_svm:.2f}%", f"{precision_svm:.2f}%"


def metrics_xgboost():
    accuracy_xgboost = xgboost_metrics['accuracy'] * 100
    f1_xgboost = xgboost_metrics['f1'] * 100
    recall_xgboost = xgboost_metrics['recall'] * 100
    precision_xgboost = xgboost_metrics['precision'] * 100
    return f"{accuracy_xgboost:.2f}%", f"{f1_xgboost:.2f}%", f"{recall_xgboost:.2f}%", f"{precision_xgboost:.2f}%"


def metrics_neural_network():
    accuracy_nn = neural_network_metrics['accuracy'] * 100
    f1_nn = neural_network_metrics['f1'] * 100
    recall_nn = neural_network_metrics['recall'] * 100
    precision_nn = neural_network_metrics['precision'] * 100
    return f"{accuracy_nn:.2f}%", f"{f1_nn:.2f}%", f"{recall_nn:.2f}%", f"{precision_nn:.2f}%"


def metrics_bert():
    accuracy_bert = bert_metrics['accuracy'] * 100
    f1_bert = bert_metrics['f1_score'] * 100
    recall_bert = bert_metrics['recall'] * 100
    precision_bert = bert_metrics['precision'] * 100
    return f"{accuracy_bert:.2f}%", f"{f1_bert:.2f}%", f"{recall_bert:.2f}%", f"{precision_bert:.2f}%"


@app.route('/')
def home():
    # Metrics
    accuracy_log, f1_log, recall_log, precision_log = logistic_regression_metrics()
    accuracy_svm, f1_svm, recall_svm, precision_svm = metrics_svm()
    accuracy_xgboost, f1_xgboost, recall_xgboost, precision_xgboost = metrics_xgboost()
    accuracy_nn, f1_nn, recall_nn, precision_nn = metrics_neural_network()
    accuracy_bert, f1_bert, recall_bert, precision_bert = metrics_bert()
    print(f'Predictions: {accuracy_bert, f1_bert, recall_bert, precision_bert}')

    return render_template('home.html',
                           accuracy_log=accuracy_log, f1_log=f1_log, recall_log=recall_log, precision_log=precision_log,
                           accuracy_svm=accuracy_svm, f1_svm=f1_svm, recall_svm=recall_svm, precision_svm=precision_svm,
                           accuracy_xgboost=accuracy_xgboost, f1_xgboost=f1_xgboost, recall_xgboost=recall_xgboost,
                           precision_xgboost=precision_xgboost, accuracy_nn=accuracy_nn, f1_nn=f1_nn,
                           recall_nn=recall_nn,
                           precision_nn=precision_nn, accuracy_bert=accuracy_bert, f1_bert=f1_bert,
                           recall_bert=recall_bert,
                           precision_bert=precision_bert)


@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    print(f'Tweet: {tweet}')

    # Preprocess the tweet and get its BERTweet embeddings
    processed_new_tweet = preprocess_tweet(tweet)
    new_tweet_embeddings = get_bertweet_embeddings(processed_new_tweet, tokenizer, model, max_length=128)

    # Ensure embeddings have correct shape for prediction
    new_tweet_embeddings = new_tweet_embeddings.reshape(1, -1)

    # Predict using the logistic regression model, SVM, and XGBoost
    prediction_lr = log_reg.predict(new_tweet_embeddings)[0]
    prediction_svm = svm_model.predict(new_tweet_embeddings)[0]
    prediction_xgboost = xgboost.predict(new_tweet_embeddings)[0]

    # For neural network, reshape to match expected input shape
    new_tweet_embeddings_nn = new_tweet_embeddings.reshape(1, new_tweet_embeddings.shape[1], 1)
    prediction_nn = neural_network.predict(new_tweet_embeddings_nn)[0]

    # Predict using the BERT model
    inputs = tokenizer(processed_new_tweet, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = bert_classifier(inputs['input_ids'], inputs['attention_mask'])

    # Get probabilities and the predicted class
    probabilities = torch.softmax(logits, dim=1)
    print(f'Probabilities: {probabilities}')
    print()
    max_index = torch.argmax(probabilities, dim=1).item()
    prediction_bert = probabilities[0][max_index].item()
    print(f'Predictions: {prediction_lr, prediction_svm, prediction_xgboost, prediction_nn, prediction_bert}')
    print(f'Index with max value: {max_index}, Max value: {prediction_bert}')

    # Metrics
    accuracy_log, f1_log, recall_log, precision_log = logistic_regression_metrics()
    accuracy_svm, f1_svm, recall_svm, precision_svm = metrics_svm()
    accuracy_xgboost, f1_xgboost, recall_xgboost, precision_xgboost = metrics_xgboost()
    accuracy_nn, f1_nn, recall_nn, precision_nn = metrics_neural_network()
    accuracy_bert, f1_bert, recall_bert, precision_bert = metrics_bert()

    # Render the template with the predictions as variables
    return render_template('home.html',
                           tweet=tweet,
                           prediction_lr=("Bullying detected" if prediction_lr == 1 else "No bullying"),
                           prediction_svm=("Bullying detected" if prediction_svm == 1 else "No bullying"),
                           prediction_xgboost=("Bullying detected" if prediction_xgboost == 1 else "No bullying"),
                           prediction_nn=("Bullying detected" if prediction_nn > 0.5 else "No bullying"),
                           prediction_bert=("Bullying detected" if prediction_bert > 0.5 else "No bullying"),
                           accuracy_log=accuracy_log, f1_log=f1_log, recall_log=recall_log, precision_log=precision_log,
                           accuracy_svm=accuracy_svm, f1_svm=f1_svm, recall_svm=recall_svm, precision_svm=precision_svm,
                           accuracy_xgboost=accuracy_xgboost, f1_xgboost=f1_xgboost, recall_xgboost=recall_xgboost,
                           precision_xgboost=precision_xgboost, accuracy_nn=accuracy_nn, f1_nn=f1_nn,
                           recall_nn=recall_nn, precision_nn=precision_nn,
                           accuracy_bert=accuracy_bert, f1_bert=f1_bert, recall_bert=recall_bert,
                           precision_bert=precision_bert)


if __name__ == '__main__':
    app.run(debug=True)
