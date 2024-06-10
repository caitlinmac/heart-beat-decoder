import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from utils import preprocess

def initialize_model():
    return RandomForestClassifier(
        random_state=101,
        n_estimators=50,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=1
    )

def main_stream():
    X_train, y_train, X_test, y_test, scaler = preprocess()
    model = initialize_model()
    model.fit(X_train, y_train)

    model_dir = 'ecg_model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(os.path.join(model_dir, 'trained_model.pkl'), 'wb') as f:
        pickle.dump((model, scaler), f)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("*** Confusion Matrix ***")
    print(confusion_matrix(y_test, y_pred))
    print("*** Classification Report ***")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

    return y_pred, y_test, model

def predict():
    y_pred, y_test, model = main_stream()
    return y_pred

if __name__ == "__main__":
    predictions = predict()
    print(predictions)
