import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import time

X_TEST = pd.DataFrame
Y_TEST = pd.Series

def clean():
    """
    Load the dataset from global variable from .env path from FOLDER_PATH and DATASET_FILE.
    Clean the dataset: removing empty columns, renaming types.
    """

    data = pd.read_csv(os.environ.get('FOLDER_PATH') + os.environ.get('DATASET_FILE'))

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

def preprocess(X = None):
    """

    Preprocess the data: cleaning, splitting, scaling and resampling.
    X: default None,
        X typically use for prediction if none it process a brand
        new data from clean().

    Returns:
    if X: None -> X_train, y_train, X_test, y_test
    if X: -> X

    The preprocess of data before training

    """
    if X is None:
        data = clean()

        # Defining the features and the target
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

    else:

        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        X = X.reshape((1,32))
        return X

def model():
    """
    Main function to preprocess data, train the model and save it into pickle and return the model.

    If the MODEL_TARGET is set to local, the model into model that function will be used
    otherwise it will runs only from the pickle.

    see load_model in main.py for more details...

    Returns: Model

    """
    X_train, y_train, X_test, y_test = preprocess()

    # The model and parameters
    model =  RandomForestClassifier(random_state=101, n_estimators=50, max_depth= None, min_samples_split= 10 , min_samples_leaf= 1)
    model.fit(X_train, y_train)

    # Uncomment to use a gridsearch
    """ # Cross-validation and hyperparameter tuning
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
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
    print("Best params: ", grid_search.best_params_)
    print("Best cross-val score: ", grid_search.best_score_)
    print(f"Accuracy: {accuracy}")
    print("*** Confusion Matrix ***")
    print(confusion_matrix(y_test, y_pred))
    print("*** Classification Report ***")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))
"""
    # Save the model every time it has been train into a pickle.
    # Save on global variable as priority if not hard path.
    model_custom_path = os.environ.get('MODEL_PICKLE_PATH')

    if model_custom_path != None:
        with (model_custom_path, "wb") as file:
            pickle.dump(model, file)
            print(f'loaded pickle from {model_custom_path}')
    else:

        with open("heartbd/models/local_model.pkl", "wb") as file:
            pickle.dump(model, file)

    return model
