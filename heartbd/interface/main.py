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
import heartbd.interface.local_model as pm
from heartbd.interface.local_model import preprocess
import os
import time



def load_model():

    model_target = os.environ.get('MODEL_TARGET')
    pickle_file_name = 'pickled'

    if model_target == 'local':
        return pm.model()
    else:
        return pickle.load(open(f"heartbd/models/{pickle_file_name}.pkl","rb"))


def sample_test():
    """
    It provides sample from data set to test the prediction that the model needs to.
    return: X_test, Y_test
    """


def predict():
    """
    Function to call the main stream for predictions.

    Returns:
    np.ndarray: Predictions from the main stream.

    """
    X, y, X_test, y_test = preprocess()
    model = load_model()

    y_pred = model.predict(X_test)



    return y_pred


if __name__ == "__main__":
    # start_time = time.time()
    #predictions = predict()
    # end_time = time.time()
    # runtime = end_time - start_time
    print(predict())
    # print(f"Runtime for predictions: {runtime} seconds")
