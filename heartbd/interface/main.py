import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import heartbd.interface.local_model as lm
from heartbd.interface.local_model import preprocess
import os
import time



def load_model():
    """
    Method: loads a model on the behavior of which the global variable has been set
    inside of .env.
    MODEL_TARGET: string,
        default:None, use 'pickle' or 'local'
        - None or 'local': Will use local_model.py
        -'pickle' : Will use the pickle saved model locally see MODEL_PICKLE for more details.
    MODEL_PICKLE: string,
        default: None,
        otherwise it has to be a folder that already exists set for the pickle path.
        <folder path + file + extension>: use a folder path and overide the default pickle path.

    .env
    >>>MODEL_TARGET='pickle'
    >>>MODEL_PICKLE_PATH='pickle_model/trained_model.pkl'
    """


    model_target = os.environ.get('MODEL_TARGET')
    model_path_pickle = os.environ.get('MODEL_PICKLE_PATH')
    pickle_file_name = 'local_pickled'

    # To load the model from the custom library MODEL_PICKLE_PATH and MODEL_TARGET has to be set properly
    if model_path_pickle != None and model_target == 'pickle':
        return pickle.load(open(model_path_pickle,"rb"))
    if model_target == 'pickle':
        return pickle.load(open(f"heartbd/models/{pickle_file_name}.pkl","rb"))
    else:
        return lm.model()



def predict(X_predict = None):
    """
    Function to call the main stream for predictions.

    Returns:
    np.ndarray: Predictions from the main stream.

    """
    # If no X_predict provided create one
    if X_predict is None:
        data = lm.clean()
        X_predict = pd.DataFrame(data.iloc[175_000, :])
        print(X_predict)
        X_predict = data.drop('type', axis = 1)

    print(f'X has a shape of {X_predict.shape}')
    X_predict = preprocess(X_predict)
    model = load_model()
    print(f"model loaded as a {os.environ.get('MODEL_TARGET')}")
    if type(model) == tuple:
        y_pred = model[0].predict(X_predict)
    else:
        y_pred = model.predict(X_predict)
    return y_pred[0]


if __name__ == "__main__":
    # start_time = time.time()
    #predictions = predict()
    # end_time = time.time()
    # runtime = end_time - start_time
    print(predict())
    # print(f"Runtime for predictions: {runtime} seconds")
