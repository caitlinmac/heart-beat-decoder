from fastapi import FastAPI
import pickle

# instantiate
app = FastAPI()

# define the root directory
@app.get('/')
def root():
	return {'greeting': 'ground control to major tom...'}

# define our prediction
# the 'main_output' variable is just a stand-in for the time being
@app.get('/predict')
def predict(main_output):
	with open('heartbd/models/main_test.pkl','rb') as file:
		model = pickle.load(file)
	prediction = model.predict(main_output)
	return {'prediction': prediction}
