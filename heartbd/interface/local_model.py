
import os
import pickle
import pandas as pd
from typing import Union
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

def clean() -> pd.DataFrame:
    """
    Load and clean the dataset.

    Uses .env variables:
    - FOLDER_PATH: Path to the folder containing the dataset.
    - DATASET_FILE: Name of the dataset file.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """

    data = pd.read_csv(os.environ.get('FOLDER_PATH') + os.environ.get('DATASET_FILE'))

    type_mapping = {
        'N'    : 'Normal',
        'SVEB' : 'Abnormal',
        'VEB'  : 'Abnormal',
        'F'    : 'Abnormal',
        'Q'    : 'Abnormal'
    }

    data         = data.drop(columns=['record'])
    data['type'] = data['type'].map(type_mapping)

    return data

def preprocess(X_predict = None) -> Union[pd.DataFrame, tuple]:
    """
    Preprocess the data for training or prediction.

    Args:
        X_predict (pd.DataFrame, optional): Defaults is None, DataFrame to preprocess for prediction.

     Returns:
        Union[pd.DataFrame, tuple]: Preprocessed data for prediction or training.
    """

    if X_predict is None:
        data = clean()

        # Defining the features and the target
        X = data.drop('type', axis=1)
        y = data['type']

        binary_type_mapping = {'Normal': 0, 'Abnormal': 1}

        y = y.map(binary_type_mapping)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Scaling the data before training
        scaler  = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Resampling and rebalancing the data
        smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Slicing to get light training memory
        subset_size = 1000
        X_train_subsample = X_train_resampled[:subset_size]
        y_train_subsample = y_train_resampled[:subset_size]

        return X_train_subsample, y_train_subsample

    else:
        scaler    = MinMaxScaler()
        X_predict = scaler.fit_transform(X_predict)
        X_predict = X_predict.reshape((1,32))
        return X_predict

def model() -> BaseEstimator:
    """
    Train and save the model.

    Returns:
        BaseEstimator: Trained model.

    """
    X_train, y_train = preprocess()

    # The model and parameters
    model =  RandomForestClassifier(random_state=101, n_estimators=50, max_depth= None, min_samples_split= 10 , min_samples_leaf= 1)
    model.fit(X_train, y_train)

    path = "heartbd/models/local_model.pkl"

    with open(path, "wb") as file:
        pickle.dump(model, file)
        print(f'saved pickle to {path}')

    return model
