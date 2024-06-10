from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd

# instantiate
app = FastAPI()

# load model from pickle file
# important to do it outside the model so that we don't have to wait for it to load
with open('heartbd/models/main_test.pkl','rb') as file:
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

# define a class for the prediction input
# this is best practice for pydantic data validation (BaseModel is from pydantic)
class InputFeatures(BaseModel):
    # define variables relevant to the incoming csv files
    _0preRR = '0_pre-RR'
    _0postRR = '0_post-RR'
    _0pPeak = '0_pPeak'
    _0tPeak = '0_tPeak'
    _0rPeak = '0_rPeak'
    _0sPeak = '0_sPeak'
    _0qPeak = '0_qPeak'
    _0qrsinterval = '0_qrs_interval'
    _0pqinterval = '0_pq_interval'
    _0qtinterval = '0_qt_interval'
    _0stinterval = '0_st_interval'
    _0qrsmorph0 = '0_qrs_morph0'
    _0qrsmorph1 = '0_qrs_morph1'
    _0qrsmorph2 = '0_qrs_morph2'
    _0qrsmorph3 = '0_qrs_morph3'
    _0qrsmorph4 = '0_qrs_morph4'
    _1preRR = '1_pre-RR'
    _1postRR = '1_post-RR'
    _1pPeak = '1_pPeak'
    _1tPeak = '1_tPeak'
    _1rPeak = '1_rPeak'
    _1sPeak = '1_sPeak'
    _1qPeak ='1_qPeak'
    _1qrsinterval = '1_qrs_interval'
    _1pqinterval = '1_pq_interval'
    _1qtinterval = '1_qt_interval'
    _1stinterval = '1_st_interval'
    _1qrsmorph0 = '1_qrs_morph0'
    _1qrsmorph1 = '1_qrs_morph1'
    _1qrsmorph2 = '1_qrs_morph2'
    _1qrsmorph3 = '1_qrs_morph3'
    _1qrsmorph4 = '1_qrs_morph4'
    # now, we assign the data types to the feature variables
    _0preRR: int
    _0postRR: int
    _0pPeak: float
    _0tPeak: float
    _0rPeak: float
    _0sPeak: float
    _0qPeak: float
    _0qrsinterval: int
    _0pqinterval: int
    _0qtinterval: int
    _0stinterval: int
    _0qrsmorph0: float
    _0qrsmorph1: float
    _0qrsmorph2: float
    _0qrsmorph3: float
    _0qrsmorph4: float
    _1preRR: int
    _1postRR: int
    _1pPeak: float
    _1tPeak: float
    _1rPeak: float
    _1sPeak: float
    _1qPeak: float
    _1qrsinterval: int
    _1pqinterval: int
    _1qtinterval: int
    _1stinterval: int
    _1qrsmorph0: float
    _1qrsmorph1: float
    _1qrsmorph2: float
    _1qrsmorph3: float
    _1qrsmorph4: float


# define the root directory of the API
@app.get('/')
def root():
	return {'greeting': 'ground control to major tom...'}

# define our prediction and call the class of InputFeatures
@app.get('/predict')
def predict(X_pred: InputFeatures):
    '''
    1. refer to all X features and store in dataframe
    2. call the model from the pickle
    3. assuming the model exists in the first place
    4. convert to np.array if needed
    5. define our prediction
    6. return a dictionary formatted as {result: <float>}
    '''
    X_pred = pd.DataFrame(locals(), index=[0])
    model = app.state.model
    assert model is not None
    X_processed = X_pred.values
    y_pred = model.predict(X_processed)
    return dict(result=float(y_pred))
