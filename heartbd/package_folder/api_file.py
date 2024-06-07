from fastapi import FastAPI
import pickle

# instantiate
app = FastAPI()

# define the root directory of the API
@app.get('/')
def root():
	return {'greeting': 'ground control to major tom...'}

# define our prediction
# the 'main_output' variable is just a stand-in for the time being
# this function loads model from pickled file and stores it in 'model'
# then sends it to the API
@app.get('/predict')
def predict(main_output):
	with open('heartbd/models/main_test.pkl','rb') as file:
		model = pickle.load(file)            # pickle-level
	prediction = model.predict(main_output)  # model-level
	return {'prediction': prediction}        # API-level

