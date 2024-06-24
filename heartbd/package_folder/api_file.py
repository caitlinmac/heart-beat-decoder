from fastapi import FastAPI
from heartbd.interface import main
import pandas as pd

# instantiate
app = FastAPI()

@app.get('/')
def root():
	return {'greeting': 'ground control to major tom...'}

# define our prediction and call the class of InputFeatures
@app.get('/predict')
def predict(_0preRR: int,
    _0postRR: int,
    _0pPeak: float,
    _0tPeak: float,
    _0rPeak: float,
    _0sPeak: float,
    _0qPeak: float,
    _0qrsinterval: int,
    _0pqinterval: int,
    _0qtinterval: int,
    _0stinterval: int,
    _0qrsmorph0: float,
    _0qrsmorph1: float,
    _0qrsmorph2: float,
    _0qrsmorph3: float,
    _0qrsmorph4: float,
    _1preRR: int,
    _1postRR: int,
    _1pPeak: float,
    _1tPeak: float,
    _1rPeak: float,
    _1sPeak: float,
    _1qPeak: float,
    _1qrsinterval: int,
    _1pqinterval: int,
    _1qtinterval: int,
    _1stinterval: int,
    _1qrsmorph0: float,
    _1qrsmorph1: float,
    _1qrsmorph2: float,
    _1qrsmorph3: float,
    _1qrsmorph4: float,
):
    '''
    Predict function to send result. The input is 32 features from ECG reading.
    Loads the trained model from the specified location and do the prediction.

    Return: predict result 0 or 1 (int)
    '''

    #Loading the model is deprecated as it loads already from the backend
    #model = app.state.model

    # pass all features into the api, feature variables indexed through the front end
    X = pd.DataFrame([_0preRR,
                    _0postRR,
                    _0pPeak,
                    _0tPeak,
                    _0rPeak,
                    _0sPeak,
                    _0qPeak,
                    _0qrsinterval,
                    _0pqinterval,
                    _0qtinterval,
                    _0stinterval,
                    _0qrsmorph0,
                    _0qrsmorph1,
                    _0qrsmorph2,
                    _0qrsmorph3,
                    _0qrsmorph4,
                    _1preRR,
                    _1postRR,
                    _1pPeak,
                    _1tPeak,
                    _1rPeak,
                    _1sPeak,
                    _1qPeak,
                    _1qrsinterval,
                    _1pqinterval,
                    _1qtinterval,
                    _1stinterval,
                    _1qrsmorph0,
                    _1qrsmorph1,
                    _1qrsmorph2,
                    _1qrsmorph3,
                    _1qrsmorph4])
    print(X.shape)
    prediction = main.predict(X)

    y_pred = int(prediction)
    return {"result": y_pred} # return a dictionary formatted as {result: <float>}
