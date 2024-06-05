import string
import joblib
import pickle
import numpy as np
import torch
import re
import pandas as pd
from emoji.core import demojize
from transformers import AutoTokenizer, AutoModel
from flask import Flask

app = Flask(__name__)

# Load the trained models
log_reg = joblib.load('pickle_file/logistic_regression.pkl')
svm_model = joblib.load('pickle_file/svm.pkl')
neural_network = joblib.load('pickle_file/neural_network.pkl')
xgboost = joblib.load('pickle_file/xgboost.pkl')
log_reg_metrics = joblib.load('pickle_file/metrics_logistic_regression.pkl')
svm_metrics = joblib.load('pickle_file/metrics_svm.pkl')
xgboost_metrics = joblib.load('pickle_file/metrics_xgboost.pkl')
neural_network_metrics = joblib.load('pickle_file/metrics_neural_network.pkl')

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


# Predict Logistic Regression
def predict_logistic_regression(tweet):
    processed_tweet = preprocess_tweet(tweet)
    tweet_embeddings = get_bertweet_embeddings(processed_tweet, tokenizer, model, max_length=128)
    tweet_embeddings = tweet_embeddings.reshape(1, -1)
    prediction = log_reg.predict(tweet_embeddings)[0]
    return prediction


# Predict SVM
def predict_svm(tweet):
    processed_tweet = preprocess_tweet(tweet)
    tweet_embeddings = get_bertweet_embeddings(processed_tweet, tokenizer, model, max_length=128)
    tweet_embeddings = tweet_embeddings.reshape(1, -1)
    prediction = svm_model.predict(tweet_embeddings)[0]
    return prediction


# Predict XGBoost
def predict_xgboost(tweet):
    processed_tweet = preprocess_tweet(tweet)
    tweet_embeddings = get_bertweet_embeddings(processed_tweet, tokenizer, model, max_length=128)
    tweet_embeddings = tweet_embeddings.reshape(1, -1)
    prediction = xgboost.predict(tweet_embeddings)[0]
    return prediction


# Predict Neural Network
def predict_neural_network(tweet):
    processed_tweet = preprocess_tweet(tweet)
    tweet_embeddings = get_bertweet_embeddings(processed_tweet, tokenizer, model, max_length=128)
    tweet_embeddings = tweet_embeddings.reshape(1, tweet_embeddings.shape[1], 1)
    prediction = neural_network.predict(tweet_embeddings)[0]
    return prediction


@app.route('/')
def hello_world():
    tweet = "You're an amazing person, and I'm grateful to have you in my life."

    # Preprocess the tweet and get its BERTweet embeddings
    processed_new_tweet = preprocess_tweet(tweet)
    new_tweet_embeddings = get_bertweet_embeddings(processed_new_tweet, tokenizer, model, max_length=128)

    # Ensure embeddings have correct shape for prediction
    new_tweet_embeddings = new_tweet_embeddings.reshape(1, -1)

    # Predict using the logistic regression model
    prediction_lr = log_reg.predict(new_tweet_embeddings)[0]
    prediction_svm = svm_model.predict(new_tweet_embeddings)[0]
    prediction_xgboost = xgboost.predict(new_tweet_embeddings)[0]

    # For neural network, reshape to match expected input shape
    new_tweet_embeddings_nn = new_tweet_embeddings.reshape(1, new_tweet_embeddings.shape[1], 1)
    prediction_nn = neural_network.predict(new_tweet_embeddings_nn)[0]

    # Return the prediction as a response
    return (f'Tweet: {tweet}\n'
            f'Logistic Regression Prediction: {"bullying detected " if prediction_lr == 1 else "no bullying"}'
            f'SVM Prediction: {"bullying detected" if prediction_svm == 1 else "no bullying"}\n'
            f'XGBoost Prediction: {"bullying detected" if prediction_xgboost == 1 else "no bullying"}\n'
            f'Neural Network Prediction: {"bullying detected" if prediction_nn > 0.5 else "no bullying"}')


if __name__ == '__main__':
    app.run(debug=True)
