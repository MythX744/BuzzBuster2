import re
import string
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from emoji import demojize
import pickle

# Load BERTweet tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModel.from_pretrained("vinai/bertweet-base")


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


# Load data
df = pd.read_csv('data/bullying.csv')
df = df.dropna()

df['label'] = df['label'].map({'bullying detected': 1, 'no bullying': 0})

# Preprocess tweets
preprocessed_tweets = [preprocess_tweet(tweet) for tweet in df["text"]]

# Get BERTweet embeddings
X = np.array([get_bertweet_embeddings(tweet, tokenizer, model, max_length=128) for tweet in preprocessed_tweets])
y = df['label']

# Save preprocessed tweets and embeddings to pickle
data_to_pickle = {
    'preprocessed_tweets': preprocessed_tweets,
    'embeddings': X,
    'labels': y
}

with open('pickle_file/preprocessed_data.pkl', 'wb') as f:
    pickle.dump(data_to_pickle, f)

print("Preprocessed tweets and embeddings saved to 'preprocessed_data.pkl'.")
