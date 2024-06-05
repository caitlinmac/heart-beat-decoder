
### THIS IS JUST THE TEMPLATE FOR WHEN WE HAVE OUR MODEL

from fastapi import FastAPI
import pickle

app = FastAPI()

@app.get('/')
def root():
    """
    define the root directory
    """
    return {'greeting': 'hello'}

@app.get('/predict')
def predict(
			# your predictions for the model etc.
			)
    """
    load the model and store it in the variable 'model'
    """
	with open('../models/model_name.pkl', 'wb') as file:
	model = pickle.load(file)

prediction = model.predict( # features to predict
							)

return {'prediction': the_prediction}
