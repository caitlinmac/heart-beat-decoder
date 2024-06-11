from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd

# instantiate
app = FastAPI()

# filename variable in case our pickle file changes
pkl_file = "local_model" # or whatever the pickle filename is

# load model from pickle file
# important to do it outside the model so that we don't have to wait for it to load
with open(f'heartbd/models/{pkl_file}.pkl','rb') as file:
    '''
    the path specified is where the file is being loaded from ('rb' is 'read binary')
    '''
    app.state.model = pickle.load(file)

# implementing FastApi middleware because it is the recommended best practice
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# define the root directory of the API
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
    our predict function in the API
    '''
    model = app.state.model # call the model from the pickle
    assert model is not None # assuming the model exists in the first place
    prediction = model.predict(_0preRR,
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
                        _1qrsmorph4,) # pass all features into the api, feature variables indexed through the front end
    y_pred = float(prediction[0])
    return {"result": y_pred} # return a dictionary formatted as {result: <float>}
