
import os
import pickle
import pandas as pd
from typing import Union
from sklearn.base import BaseEstimator
import heartbd.interface.local_model as lm
from heartbd.interface.local_model import preprocess

def load_model() -> Union[BaseEstimator, tuple]:
    """
    Loads the model based on the configuration set in .env.

    Uses .env variables:
    - MODEL_TARGET: The method used to load the model.
    - MODEL_PICKLE_PATH: The path of the pickle file.

    Returns:
        model: The loaded BaseEstimator model.
    """

    model_target = os.environ.get('MODEL_TARGET')
    model_path_pickle = os.environ.get('MODEL_PICKLE_PATH')
    pickle_file_name = 'local_pickled'

    # To load the model from the custom library
    # MODEL_PICKLE_PATH and MODEL_TARGET has to be set properly
    try:
        if model_target == 'pickle':
            if model_path_pickle:
                with open(model_path_pickle, 'rb') as file:
                    return pickle.load(file)
            else:
                with open(f"heartbd/models/{pickle_file_name}.pkl", 'rb') as file:
                    return pickle.load(file)
        else:
            return lm.model()
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

def predict(X_predict = None) -> int:
    """
    Makes a prediction using the specified model.

    Args:
        X_predict (pd.DataFrame, optional): Defaults is None, DataFrame to make predictions on.

    Uses .env variables:
    - MODEL_TARGET: The method used to load the model.

    Returns:
        int: Prediction result.

    """
    # If no X_predict provided create one
    if X_predict is None:

        data = lm.clean()
        X_predict = pd.DataFrame(data.iloc[175000, :])
        X_predict = X_predict.drop('type')

    X_predict = preprocess(X_predict)
    model = load_model()

    print(f'X has a shape of {X_predict.shape}')
    print(f"model loaded as {os.environ.get('MODEL_TARGET')}")

    try:
        y_pred = model[0].predict(X_predict)
        return y_pred[0]
    except Exception as e:
        raise Exception(f"Error during prediction: {e}")

if __name__ == "__main__":
    try:
        result = predict()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Prediction error: {e}")
