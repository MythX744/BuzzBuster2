import torch
from torch import nn
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import pandas as pd

# Load the dataset
data = pd.read_csv('data/bullying.csv')
data['label'] = data['label'].apply(lambda x: 1 if x == 'bullying detected' else 0)
X = data['text']
y = data['label']

# Tokenize the text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_text(texts, labels, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []
    labels = torch.tensor(np.array(labels))  # Convert to numpy array first

    for text in texts:
        encoding = tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks, labels


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the training and validation sets
X_val_list = X_val.astype(str).tolist()
train_input_ids, train_attention_masks, train_labels = tokenize_text(X_train, y_train, tokenizer)
val_input_ids, val_attention_masks, val_labels = tokenize_text(X_val_list, y_val, tokenizer)


# Create TensorDatasets
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)


# Create DataLoaders
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


# Define your BERT-based classifier model
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


# Create the model and move it to the appropriate device
model = BERTClassifier('bert-base-uncased', num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define the optimizer and scheduler
num_epochs = 4
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in train_dataloader:
        model.train()
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
    # Evaluation
    model.eval()
    predictions = []
    actual_labels = []
    for batch in val_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        predictions.extend(preds.cpu().tolist())
        actual_labels.extend(labels.cpu().tolist())
    accuracy = accuracy_score(actual_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predictions, average='binary')
    report = classification_report(actual_labels, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)


# Save the model
joblib.dump(model, 'bert_classifier.pkl')

# Save metrics
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
}
joblib.dump(metrics, 'metrics_bert_classifier.pkl')

