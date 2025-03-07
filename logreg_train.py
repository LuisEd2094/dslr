import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, alpha=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    for _ in range(epochs):
        z = X.dot(weights)
        h = sigmoid(z)
        gradient = (1/m) * X.T.dot(h - y)
        weights -= alpha * gradient
    return weights

def main():
    # Load data
    df = pd.read_csv('dataset_train.csv')
    
    # Drop irrelevant columns
    df = df.drop(['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1)
    
    # Encode target variable ("Hogwarts House")
    le = LabelEncoder()
    y = le.fit_transform(df['Hogwarts House'])
    
    # Separate features and target
    X = df.drop('Hogwarts House', axis=1)
    
    # Fill missing numeric values with mean
    numeric_cols = X.select_dtypes(include=np.number).columns
    feature_means = X[numeric_cols].mean()
    X[numeric_cols] = X[numeric_cols].fillna(feature_means)
    
    # Standardize features
    feature_stds = X[numeric_cols].std()
    X[numeric_cols] = (X[numeric_cols] - feature_means) / feature_stds
    
    # Split into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(X.values, y, test_size=0.2, random_state=42)
    
    # Train One-vs-All classifiers
    houses = le.classes_
    weights_dict = {}
    for idx, house in enumerate(houses):
        y_binary = np.where(y_train == idx, 1, 0)
        weights = train_logistic_regression(X_train, y_binary)
        weights_dict[house] = weights
    
    # Validate on the validation set
    val_probs = np.zeros((X_val.shape[0], len(houses)))
    for i, house in enumerate(houses):
        z = X_val.dot(weights_dict[house])
        val_probs[:, i] = sigmoid(z)
    val_preds = [houses[np.argmax(row)] for row in val_probs]
    val_accuracy = accuracy_score(le.inverse_transform(y_val), val_preds)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    # Save model parameters
    np.savez('weights.npz', 
             weights=weights_dict, 
             feature_means=feature_means.to_numpy(), 
             feature_means_index=feature_means.index.tolist(), 
             feature_stds=feature_stds,
             label_encoder=le.classes_)

if __name__ == '__main__':
    main()