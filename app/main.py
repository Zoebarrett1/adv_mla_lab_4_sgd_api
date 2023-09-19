from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

# Instantiate a FastAPI() class and save it into a variable called app
app = FastAPI()

# Load trained model and save to a variable called sgd_pipe
sgd_pipe = load('../models/sgd_pipeline.joblib')

# Create read_root function
@app.get("/")
def read_root():
    """
    Function that will return a dictionary with Hello as key and World as value.

    Parameters
    ----------
    no parameters

    Returns
    -------
    dictionary
        'Hello' as key and 'World' as value.
    """
    return {"Hello": "World"}

# Create healthcheck function 
@app.get('/health', status_code=200)
def healthcheck():
    """
    Function that will return a dictionary with Hello as key and World as value.

    Parameters
    ----------
    no parameters

    Returns
    -------
    string
        'SGDClassifier is all ready to go!'
    """
    return 'SGDClassifier is all ready to go!'

# Create format_features function 
def format_features(
    general_health: str,
    checkup: str,
    exercise: str,
    heart_disease: str,
    skin_cancer: str,
    other_cancer: str,
    depression: str,
    diabetes: str,
    arthritis: str,
    sex: str,
    age_category: str,
    height: float,
    weight: float,
    bmi: float,
    smoking_history: str,
    alcohol_consumption: float,
    fruit_consumption: float,
    green_vegetables_consumption: float,
    friedpotato_consumption: float,
    ):
    """
    Function that will return a dictionary with Hello as key and World as value.

    Parameters
    ----------
    12x strings, 7x floats
        Collection individuals demographic and health data
    
    Returns
    -------
    dictionary
        Names of the features as keys and the input parameters as lists
    """
    return {
        'General_Health': [general_health],
        'Checkup': [checkup],
        'Exercise': [exercise],
        'Heart_Disease': [heart_disease],
        'Skin_Cancer': [skin_cancer],
        'Other_Cancer': [other_cancer],
        'Depression': [depression],
        'Diabetes': [diabetes],
        'Arthritis': [arthritis],
        'Sex': [sex],
        'Age_Category': [age_category],
        'Height_(cm)': [height],
        'Weight_(kg)': [weight],
        'BMI': [bmi],
        'Smoking_History': [smoking_history],
        'Alcohol_Consumption': [alcohol_consumption],
        'Fruit_Consumption': [fruit_consumption],
        'Green_Vegetables_Consumption': [green_vegetables_consumption],
        'FriedPotato_Consumption': [friedpotato_consumption]
    }

# Create predict function
@app.get("/cvd/risks/prediction")
def predict(
    general_health: str,
    checkup: str,
    exercise: str,
    heart_disease: str,
    skin_cancer: str,
    other_cancer: str,
    depression: str,
    diabetes: str,
    arthritis: str,
    sex: str,
    age_category: str,
    height: float,
    weight: float,
    bmi: float,
    smoking_history: str,
    alcohol_consumption: float,
    fruit_consumption: float,
    green_vegetables_consumption: float,
    friedpotato_consumption: float,
):
    """
    Function that will format the input parameters as dictionary, convert it to a dataframe and make prediction with sgd_pipe.

    Parameters
    ----------
    12x strings, 7x floats
        Collection individuals demographic and health data
    
    Returns
    -------
    list
        A list containing a prediction based on the inputs
    """
    features = format_features(
        general_health,
        checkup,
        exercise,
        heart_disease,
        skin_cancer,
        other_cancer,
        depression,
        diabetes,
        arthritis,
        sex,
        age_category,
        height,
        weight,
        bmi,
        smoking_history,
        alcohol_consumption,
        fruit_consumption,
        green_vegetables_consumption,
        friedpotato_consumption
        )
    obs = pd.DataFrame(features)
    pred = sgd_pipe.predict(obs)
    return JSONResponse(pred.tolist())
