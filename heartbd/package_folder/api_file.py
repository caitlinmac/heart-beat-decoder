
### THIS IS JUST THE TEMPLATE FOR WHEN WE HAVE OUR MODEL

from fastapi import FastAPI
import pickle

# instantiate
app = FastAPI()

# define the root directory
@app.get('/')
def root():
	return {'greeting': 'hello'}

# define our prediction
@app.get('/predict')
def predict(main_output):
	with open('heartbd/models/main_test.pkl','rb') as file:
		model = pickle.load(file)
	prediction = model.predict(main_output)
	return {'prediction': prediction}
