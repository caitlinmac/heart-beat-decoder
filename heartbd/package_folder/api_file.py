from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle


# instantiate
app = FastAPI()

# implementing middleware because it is best practice
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# define the root directory of the API
@app.get('/')
def root():
	return {'greeting': 'ground control to major tom...'}

 # define our prediction
    # the 'main_output' variable is just a stand-in for the time being
    # this function loads model from pickled file and stores it in 'model'
    # then sends it to the API
0preRR = '0_pre-RR'
0postRR = '0_post-RR'
0pPeak = '0_pPeak'
0tPeak = '0_tPeak'
0rPeak = '0_rPeak'
0sPeak = '0_sPeak'
0qPeak = '0_qPeak'
0qrsinterval = '0_qrs_interval'
0pqinterval = '0_pq_interval'
0qtinterval = '0_qt_interval'
0stinterval = '0_st_interval'
0qrsmorph0 = '0_qrs_morph0'
0qrsmorph1 = '0_qrs_morph1'
0qrsmorph2 = '0_qrs_morph2'
0qrsmorph3 = '0_qrs_morph3'
0qrsmorph4 = '0_qrs_morph4'
1preRR = '1_pre-RR'
1postRR = '1_post-RR'
1pPeak = '1_pPeak'
1tPeak = '1_tPeak'
1rPeak = '1_rPeak'
1sPeak = '1_sPeak'
1qPeak ='1_qPeak'
1qrsinterval = '1_qrs_interval'
1pqinterval = '1_pq_interval'
1qtinterval = '1_qt_interval'
1stinterval = '1_st_interval'
1qrsmorph0 = '1_qrs_morph0'
1qrsmorph1 = '1_qrs_morph1'
1qrsmorph2 = '1_qrs_morph2'
1qrsmorph3 = '1_qrs_morph3'
1qrsmorph4 = '1_qrs_morph4'

class PredictionInput(BaseModel):
'''

'''
    0preRR: int
    0postRR: int
    0pPeak: float
    0tPeak: float
    0rPeak: float
    0sPeak: float
    0qPeak: float
    0qrsinterval: int
    0pqinterval: int
    0qtinterval: int
    0stinterval: int
    0qrsmorph0: float
    0qrsmorph1: float
    0qrsmorph2: float
    0qrsmorph3: float
    0qrsmorph4: float
    1preRR: int
    1postRR: int
    1pPeak: float
    1tPeak: float
    1rPeak: float
    1sPeak: float
    1qPeak: float
    1qrsinterval: int
    1pqinterval: int
    1qtinterval: int
    1stinterval: int
    1qrsmorph0: float
    1qrsmorph1: float
    1qrsmorph2: float
    1qrsmorph3: float
    1qrsmorph4: float




@app.get('/predict')
def predict('0_pre-RR': dtype('int64'),
            '0_post-RR': dtype('int64'),
            '0_pPeak': dtype('float64'),
            '0_tPeak': dtype('float64'),
            '0_rPeak': dtype('float64'),
            '0_sPeak': dtype('float64'),
            '0_qPeak': dtype('float64'),
            '0_qrs_interval': dtype('int64'),
            '0_pq_interval': dtype('int64'),
            '0_qt_interval': dtype('int64'),
            '0_st_interval': dtype('int64'),
            '0_qrs_morph0': dtype('float64'),
            '0_qrs_morph1': dtype('float64'),
            '0_qrs_morph2': dtype('float64'),
            '0_qrs_morph3': dtype('float64'),
            '0_qrs_morph4': dtype('float64'),
            '1_pre-RR': dtype('int64'),
            '1_post-RR': dtype('int64'),
            '1_pPeak': dtype('float64'),
            '1_tPeak': dtype('float64'),
            '1_rPeak': dtype('float64'),
            '1_sPeak': dtype('float64'),
            '1_qPeak': dtype('float64'),
            '1_qrs_interval': dtype('int64'),
            '1_pq_interval': dtype('int64'),
            '1_qt_interval': dtype('int64'),
            '1_st_interval': dtype('int64'),
            '1_qrs_morph0': dtype('float64'),
            '1_qrs_morph1': dtype('float64'),
            '1_qrs_morph2': dtype('float64'),
            '1_qrs_morph3': dtype('float64'),
            '1_qrs_morph4': dtype('float64')
            ):
	# with open('heartbd/models/main_test.pkl','rb') as file:
	# 	model = pickle.load(file)            # pickle-level
	# prediction = model.predict(main_output)  # model-level
	# return {'prediction': prediction}        # API-level
