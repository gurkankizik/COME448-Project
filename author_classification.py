import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
from joblib import Parallel, delayed

# Load texts and labels from folder
def load_texts_from_folder(folder_path):
    texts = []
    labels = []
    for author in os.listdir(folder_path):
        author_folder = os.path.join(folder_path, author)
        if os.path.isdir(author_folder):
            for file_name in os.listdir(author_folder):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(author_folder, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        texts.append(file.read())
                        labels.append(author)
    return texts, labels

# Extract embeddings using BERT with batch processing
def extract_bert_embeddings(texts, tokenizer, model, max_length=512, batch_size=16, device=None):
    embeddings = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**encodings)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings).cpu().numpy()

# Train a classifier
def train_classifier(model_name, X_train, y_train):
    if model_name == 'SVM':
        model = SVC(kernel='linear', cache_size=1000)  # Increase cache size for faster SVM
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # Parallelize random forest
    elif model_name == 'NaiveBayes':
        model = MultinomialNB()
    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif model_name == 'XGBoost':
        model = XGBClassifier(n_jobs=-1)  # Parallelize XGBoost
    elif model_name == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=(100,), n_jobs=-1)  # Parallelize MLP
    else:
        raise ValueError("Unsupported model")
    
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Main function
def main():
    # Dataset path
    dataset_path = r"dataset_authorship"
    print("Loading data...")
    texts, labels = load_texts_from_folder(dataset_path)

    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42)

    # Load BERT tokenizer and model
    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)

    # Extract BERT embeddings for training and testing data in parallel
    print("Extracting BERT embeddings for training data...")
    X_train_embeddings = extract_bert_embeddings(X_train, tokenizer, bert_model, device=device)
    print("Extracting BERT embeddings for testing data...")
    X_test_embeddings = extract_bert_embeddings(X_test, tokenizer, bert_model, device=device)

    # Train the model
    model_name = 'SVM'  # Choose the model type
    print(f"Training {model_name} model...")
    model = train_classifier(model_name, X_train_embeddings, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    accuracy, precision, recall, f1 = evaluate_model(model, X_test_embeddings, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()