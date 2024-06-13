"""

This core module makes every operation in order to set and get a local_model from scikit-learn.
To set the model locally use this module to train then it is saved into a pickle.
List of .env variable used:
-FOLDER_PATH: Folder path of the csv dataset.
-DATASET_FILE: File name and extension of dataset.

"""


import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import os



def clean():
    """
    Load the dataset and use global variable from .env path from FOLDER_PATH and DATASET_FILE.
    Clean the dataset: removing empty columns and remaps types.

    (.env)
    >>>FOLDER_PATH='raw_data/'
    >>>DATASET_FILE='heart_readings.csv'

    Returns:
        pandas.DataFrame
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

def preprocess(X_predict = None):
    """
    The preprocess of data before training

    Preprocess the data: cleaning, splitting, scaling and resampling.
    X_predict (pd.DataFrame): default = None,
        X_predict is typically use for prediction if X is not provided then processes a brand
        new data from clean().

    Returns:
    if X_predict: None -> X_train, y_train, X_test, y_test
    if X_predict: -> numpy.ndarray

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

        return X_train_subsample, y_train_subsample, X_test, y_test

    else:

        scaler    = MinMaxScaler()
        X_predict = scaler.fit_transform(X_predict)
        X_predict = X_predict.reshape((1,32))

        return X_predict

def model():
    """
    Main function of this module is used to preprocess data,
    train the model, save it into pickle into a default path
    and returns the model.

    Returns: scikit-learn.model

    """
    X_train, y_train, X_test, y_test = preprocess()

    # The model and parameters
    model =  RandomForestClassifier(random_state=101, n_estimators=50, max_depth= None, min_samples_split= 10 , min_samples_leaf= 1)
    model.fit(X_train, y_train)

    # Uncomment to use a gridsearch
    # Cross-validation and hyperparameter tuning
    # cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }

    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy')
    # grid_search.fit(X_train, y_train)

    # best_model = grid_search.best_estimator_
    # y_pred = best_model.predict(X_test)

    # accuracy = accuracy_score(y_test, y_pred)
    # # Comment out the print statements
    # print("Best params: ", grid_search.best_params_)
    # print("Best cross-val score: ", grid_search.best_score_)
    # print(f"Accuracy: {accuracy}")
    # print("*** Confusion Matrix ***")
    # print(confusion_matrix(y_test, y_pred))
    # print("*** Classification Report ***")
    # print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

    # Save the model every time it has been train into a pickle.
    # Save on global variable as priority if not hard path.


    path = "heartbd/models/local_model.pkl"

    with open(path, "wb") as file:
        pickle.dump(model, file)
        print(f'saved pickle to {path}')

    return model
