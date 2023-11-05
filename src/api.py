#import library
import pandas as pd
import utils as utils
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from scaling import predict_score
from preprocessing import transform_woe

class ApiData(BaseModel):
    Age : int
    Income : int
    LoanAmount : int
    MonthsEmployed : int
    NumCreditLines : int
    InterestRate : float
    LoanTerm : int
    DTIRatio : float
    Education : str
    EmploymentType : str
    MaritalStatus : str 
    HasMortgage : str 
    HasDependents : str 
    LoanPurpose : str 
    HasCoSigner : str 

config_data = utils.config_load()

app = FastAPI()

@app.get('/')
def home():
    """Endpoint to check if the API is running."""
    return "Hello world"

@app.post('/predict')
def get_data(data: ApiData):
    """Endpoint to predict credit approval status based on input data.

    Args
    ----
    data (ApiData): Input data for credit approval prediction.

    Returns
    -------
    dict: Response containing credit score, probability, and recommendation.
    """
    #load columns list for the input
    columns_ = config_data['columns_']
    list_input = [
        data.Age, data.Income, data.LoanAmount, 
        data.MonthsEmployed, data.NumCreditLines, data.InterestRate, 
        data.LoanTerm, data.DTIRatio, data.Education, 
        data.EmploymentType, data.MaritalStatus,
        data.HasMortgage, data.HasDependents, data.LoanPurpose,
        data.HasCoSigner
    ]

    input_table = pd.DataFrame({'0': list_input}, index=columns_).T

    y_score = predict_score(raw_data=input_table, config_data=config_data)
    
    best_model = utils.pickle_load(config_data['best_model_path'])
    input_woe = transform_woe(raw_data=input_table, config_data=config_data)
    y_prob = best_model.predict_proba(input_woe)[0][0]
    y_prob = round(y_prob, 2)

    cutoff_score = config_data['cutoff_score']
    if y_score > cutoff_score:
        y_status = "APPROVE"
    else:
        y_status = "REJECT"

    #package the results into a response object
    results = {
        'Score': y_score,
        'Proba': y_prob,
        'Recommendation': y_status
    }

    return results


if __name__ == '__main__':
    uvicorn.run('api:app',
                host = '127.0.0.1',
                port = 8000)
