"""
Takes a model based on the option set inside of .env and predict from that model.
The core function is predict(X_predict) which runs all the modules and local_model.py
List of .env variable used:
-MODEL_TARGET:: Folder path of the csv dataset.
-MODEL_PICKLE: File name and extension of dataset.
"""

import pandas as pd
import os
import pickle
import heartbd.interface.local_model as lm
from heartbd.interface.local_model import preprocess
import os



def load_model():
    """
    Method: loads a model on the behavior of which the global variable has been set
    inside of .env.

    (.env)
    ### local  : Use model from scratch and dataset as well.
    ### pickle : Use model from a saved file already trained.
    >>>MODEL_TARGET='pickle'
    >>>MODEL_PICKLE_PATH='pickle_model/trained_model.pkl'

    Returns:
        scikit-learn.model
    """


    model_target      = os.environ.get('MODEL_TARGET')
    model_path_pickle = os.environ.get('MODEL_PICKLE_PATH')
    pickle_file_name  = 'local_pickled'

    # To load the model from the custom library MODEL_PICKLE_PATH and MODEL_TARGET has to be set properly
    if model_path_pickle != None and model_target == 'pickle':
        return pickle.load(open(model_path_pickle,"rb"))

    if model_target == 'pickle':
        return pickle.load(open(f"heartbd/models/{pickle_file_name}.pkl","rb"))
    else:
        return lm.model()



def predict(X_predict = None):
    """
    Make a prediction with a provided test. If none is provided there will be one created by default.
    The model used depends on the MODEL_TARGET. Which is typically local or pickle.

    Returns:
        int : Predictions from the model used.

    """
    # If no X_predict provided create one
    if X_predict is None:

        data      = lm.clean()
        X_predict = pd.DataFrame(data.iloc[175000, :])

        print(X_predict)

        X_predict = X_predict.drop('type')


    X_predict = preprocess(X_predict)
    model = load_model()

    print(f'X has a shape of {X_predict.shape}')
    print(f"model loaded as {os.environ.get('MODEL_TARGET')}")

    if type(model) == tuple:
        y_pred = model[0].predict(X_predict)
    else:
        y_pred = model.predict(X_predict)
    return y_pred[0]

if __name__ == "__main__":
    print(predict())
