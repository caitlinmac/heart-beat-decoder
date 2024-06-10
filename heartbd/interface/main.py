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

    model_target = os.environ.get('MODEL_TARGET')
    model_path_pickel = os.environ.get('MODEL_PICKEL_PATH')
    pickle_file_name = 'local_pickled'

    if model_path_pickel != None and model_target == 'pickle':
        return pickle.load(open(model_path_pickel,"rb"))
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
    if X_predict is None:
        data = lm.clean()

        fraction_test = 0.2
        slice = int(data.shape[0] *fraction_test // 1)
        X_predict = data.iloc[:slice, :]
        X_predict = preprocess(X_predict)
        print(X_predict.shape)

    X, y, X_test, y_test = preprocess()
    print(f'x has a shape of {X_test.shape}')
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
