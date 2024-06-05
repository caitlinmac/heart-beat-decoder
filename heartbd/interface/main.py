
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

#######################################################################################

def test():
    return 21 == 21

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
    Put descriptions please of what this function is doing
    """
    data = load_data('raw_data/','INCART 2-lead Arrhythmia Database.csv')

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
    Good practice for any further modification
    """
    return RandomForestClassifier(random_state=101, n_estimators=50)



def main_stream():

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

# get file name without extension and store as 'filename'
filepath = os.path.basename(__file__) # with extension
filename, _ = os.path.splitext(filepath)

# export file to a pickle file as 'filename_test.pkl' and
with open(f'heartbd/models/{filename}_test.pkl','wb') as file:
    pickle.dump(predict(), file)
    print(f'Model is successfully saved as "{filename}_test.pkl"! Consider it pickled.')
