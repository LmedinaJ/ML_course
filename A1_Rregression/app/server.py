from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import pickle

### to replace blank values in the form 
replace_values = {
    'car_old':11,
    'engine':1387.86,
    'max_power':91,
    'transmission':0,
    'owner':1,
    'km_driven':52238.81,
    'fuel':0
}

# Load the model from disk
filename = 'app/car-price.model'
loaded_model = pickle.load(open(filename, 'rb'))

# load the scaler
with open("app/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

templates =  Jinja2Templates(directory='app/templates')

# Define the Pydantic model for input validation
class PredictionRequest(BaseModel):
    features: list[float]  # must be a list of floats

app = FastAPI()

@app.get('/')
async def read_root(request: Request):
    return templates.TemplateResponse('index.html',
                                      {'request':request,
                                       'name':'testeo'})
    #return {'message': 'Model for car price predictions'}

@app.post("/submit")
def handle_form(car_year = Form(...), engine = Form(...),
                max_power = Form(...), transmission = Form(...),
                owner = Form(...), km_driven = Form(...),
                fuel = Form(...)):
    
    if car_year == 0 or car_year == '':
        car_year = replace_values['car_old']
    if engine == 0  or engine == '':
        engine = replace_values['engine']
    if max_power == 0  or max_power == '':
        max_power = replace_values['max_power']
    if transmission == 0  or transmission == '':
        transmission = replace_values['transmission']
    if owner == 0  or owner == '':
        owner = replace_values['owner']
    if km_driven == 0  or km_driven == '':
        km_driven = replace_values['km_driven']
    if fuel == 0  or fuel == '':
        fuel = replace_values['fuel']
    
    values = np.array([[2025 - car_year, np.log(engine), max_power, transmission, owner, np.log(km_driven), fuel]])
    values = scaler.transform(values)
    predicted_price_car = loaded_model.predict(values)

    return {"car_price": np.round(np.exp(predicted_price_car)).item()}

@app.post('/predict')
def predict(data: PredictionRequest):
    try:
        # Extract features from the request
        features = np.array(data.features).reshape(1, -1)

        # Make prediction
        predicted_price_car = loaded_model.predict(features)

        # Return the prediction (apply transformations if needed, e.g., exponentiation)
        return {'predicted_value': np.round(np.exp(predicted_price_car)).item()}
    except Exception as e:
        return {'error': str(e)}

