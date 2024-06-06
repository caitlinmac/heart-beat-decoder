

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



def test():
    return 21 == 21


def load_data(folder_path, folder_name) -> pd.core.frame.DataFrame:
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

def clean()-> pd.core.frame.DataFrame:
    """
    It cleans the dataset before preprocessing features
    """
    
    data = load_data(os.environ.get('FOLDER_PATH'), os.environ.get('DATASET_FILE'))

    type_names = {
        'N': 'Normal',
        'SVEB': 'Supraventricular ectopic beat',
        'VEB': 'Ventricular ectopic beat',
        'F': 'Fusion beat',
        'Q': 'Unknown beat'
    }

    data = data.drop(columns=['record'])

    return data


def preprocess():
    """
    The preprocess of data before training
    return X_train, y_train, X_test, y_test
    """
    data = clean()

    X = data.drop('type', axis=1)
    y = data['type']
    type_mapping = {'N': 0, 'SVEB': 1, 'VEB': 2, 'F': 3, 'Q': 4}
    y = y.map(type_mapping)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scaling the data before training
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Resampling and rebalance the data
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    #Slicing to get light training memory
    subset_size = 10000
    X_train_subsample = X_train_resampled[:subset_size]
    y_train_subsample = y_train_resampled[:subset_size]

    return X_train_subsample, y_train_subsample, X_test, y_test


def initialize_model():
    """
    Initialize the choosen model
    """
    return RandomForestClassifier(random_state=101, n_estimators=50)

def main_stream():
    """
    This has the structure to load every step one by one.

    """
    X_train, y_train, X_test, y_test = preprocess()

    model = initialize_model()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    #accuracy = accuracy_score(y_test, y_pred)
    return y_pred

def predict():
    return main_stream()


main_output = predict()
main_output # essentially returns y_pred in a really convoluted way

################################ EXPORT FILE ###################################

# refer to file in an error-safe way
try:
    file_path = __file__
except NameError:
    file_path = 'main.py'

# parse file name and extension and store file name as 'filename' variable
filepath = os.path.basename(__file__)
filename, _ = os.path.splitext(filepath)

# export file to a pickle file as 'filename_pickled.pkl' and
with open(f'heartbd/models/{filename}_pickled.pkl','wb') as file:
    pickle.dump(main_output, file)
    print(f'The model is successfully saved as "{filename}_pickled.pkl"! Consider <{filename}{ _}> pickled.')
=======

def test():
    print(predict())

if __name__ == "__main__":
    test()

