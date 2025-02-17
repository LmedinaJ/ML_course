from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import dill
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
filename_rf = 'car-price.model'
rf_model = pickle.load(open(filename_rf, 'rb'))
filename_poly = 'car-price-poly-mini-lr-001-momentum03-xavier-false-dill.model'
poly_model = dill.load(open(filename_poly, 'rb'))#pickle.load(open(filename_poly, 'rb'))

# load the scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

templates =  Jinja2Templates(directory='templates')

# Define the Pydantic model for input validation
class PredictionRequest(BaseModel):
    features: list[float]  # must be a list of floats

app = FastAPI()

@app.get('/')
async def read_root(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post("/submit")
async def handle_form(
    request: Request,
    car_year: str = Form(...),
    engine: str = Form(...),
    max_power: str = Form(...),
    transmission: str = Form(...),
    owner: str = Form(...),
    km_driven: str = Form(...),
    fuel: str = Form(...),
    model_choice: str = Form(...)
):
    def to_float_or_default(value: str, default: float):
        return float(value.strip()) if value.strip() != "" else default
    
    car_year = to_float_or_default(car_year, replace_values['car_old'])
    engine = to_float_or_default(engine, replace_values['engine'])
    max_power = to_float_or_default(max_power, replace_values['max_power'])
    transmission = to_float_or_default(transmission, replace_values['transmission'])
    owner = to_float_or_default(owner, replace_values['owner'])
    km_driven = to_float_or_default(km_driven, replace_values['km_driven'])
    fuel = to_float_or_default(fuel, replace_values['fuel'])


    # Prepare input features (example transformation)
    values = np.array([[2025 - car_year, np.log(engine), max_power, transmission, owner, np.log(km_driven), fuel]])
    values = scaler.transform(values)

    # Use the model_choice to select which model to use.
    if model_choice == "random_forest":
        # Use RandomForest model 
        predicted_price_car = rf_model.predict(values)
    if model_choice == "polynomial_regression":
        # Use polynomial regression model 
        predicted_price_car = poly_model.predict(values)
    
    # Calculate final car price prediction (applying inverse transformation if needed)
    predicted_price = np.round(np.exp(predicted_price_car)).item()

    # Render the same page (index.html) and pass the predicted price to the template.
    return templates.TemplateResponse("index.html", {"request": request, "car_price": predicted_price})

@app.post('/predict')
def predict(data: PredictionRequest):
    try:
        # Extract features from the request
        features = np.array(data.features).reshape(1, -1)

        # Make prediction
        predicted_price_car = poly_model.predict(features)

        # Return the prediction (apply transformations if needed, e.g., exponentiation)
        return {'predicted_value': np.round(np.exp(predicted_price_car)).item()}
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=80, reload=True)