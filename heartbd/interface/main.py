import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import time


def load_data(folder_path, folder_name):
    """
    folder_path: string,
                 path of which folder has been saved on project
                 (folder_path = 'raw_data/')

    folder_name: string,
                 name of csv file inside of folder.

           Method:
               load_data('raw_data/Arythmia_monitor.csv') return (pandas.core.frame.DataFrame)
    """

    return pd.read_csv(folder_path + folder_name)


def clean():
    """
    Clean the dataset: removing empty columns, renaming types.
    """
    data = load_data(os.environ.get('FOLDER_PATH'), os.environ.get('DATASET_FILE'))

    type_mapping = {
        'N': 'Normal',
        'SVEB': 'Abnormal',
        'VEB': 'Abnormal',
        'F': 'Abnormal',
        'Q': 'Abnormal'
    }

    data = data.drop(columns=['record'])
    data['type'] = data['type'].map(type_mapping)
    return data


def preprocess():
    """
    Preprocess the data: cleaning, splitting, scaling and resampling.

    Returns:
    tuple: Processed training and testing data.
    """
    data = clean()

    X = data.drop('type', axis=1)
    y = data['type']
    binary_type_mapping = {'Normal': 0, 'Abnormal': 1}
    y = y.map(binary_type_mapping)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scaling the data before training
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Resampling and rebalancing the data
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Slicing to get light training memory
    subset_size = 1000
    X_train_subsample = X_train_resampled[:subset_size]
    y_train_subsample = y_train_resampled[:subset_size]

    return X_train_subsample, y_train_subsample, X_test, y_test


def initialize_model():
    """
    Initialize the machine learning model.

    Returns:
    RandomForestClassifier.
    """
    return RandomForestClassifier(random_state=101, n_estimators=50)


def main_stream():
    """
    Main function to preprocess data, train the model, and make predictions.

    Returns:
    np.ndarray: Predictions from the model.
    """
    X_train, y_train, X_test, y_test = preprocess()

    model = initialize_model()

    # Cross-validation and hyperparameter tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Comment out the print statements
    # print("Best params: ", grid_search.best_params_)
    # print("Best cross-val score: ", grid_search.best_score_)
    # print(f"Accuracy: {accuracy}")
    # print("*** Confusion Matrix ***")
    # print(confusion_matrix(y_test, y_pred))
    # print("*** Classification Report ***")
    # print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

    return y_pred, y_test, best_model


def predict():
    """
    Function to call the main stream for predictions.

    Returns:
    np.ndarray: Predictions from the main stream.
    """
    y_pred, y_test, model = main_stream()
    return y_pred


if __name__ == "__main__":
    # start_time = time.time()
    predictions = predict()
    # end_time = time.time()
    # runtime = end_time - start_time
    print(predictions)
    # print(f"Runtime for predictions: {runtime} seconds")
