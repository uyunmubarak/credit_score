#import library
import utils as utils
import pandas as pd
import numpy as np

#function to convert the model's output into score points
def scaling():
    """
    Assign score points to each attribute based on the logistic regression model's coefficients.

    This function calculates the score points for each attribute using the logistic regression model's coefficients,
    reference score, reference odds, and PDO (Points to Double the Odds) provided in the configuration data.
    The resulting score points are saved in a table and dumped as a pickle file.

    Returns
    -------
    pd.DataFrame: A table containing attribute names, WOE values, and calculated score points.
    """

    #define the references: score, odds, pdo
    pdo = config_data['pdo']
    score = config_data['score_ref']
    odds = config_data['odds_ref']

    #load the best model
    best_model_path = config_data['best_model_path']
    best_model = utils.pickle_load(best_model_path)

    #load the WOE table
    WOE_table_path = config_data['WOE_table_path']
    WOE_table = utils.pickle_load(WOE_table_path)

    #load the best model's estimates table
    best_model_summary_path = config_data['best_model_summary_path']
    best_model_summary = utils.pickle_load(best_model_summary_path)

    #calculate Factor and Offset
    factor = pdo/np.log(2)
    offset = score-(factor*np.log(odds))

    print('===================================================')
    print(f"Odds of good of {odds}:1 at {score} points score.")
    print(f"{pdo} PDO (points to double the odds of good).")
    print(f"Offset = {offset:.2f}")
    print(f"Factor = {factor:.2f}")
    print('===================================================')

    #define n = number of characteristics
    n = best_model_summary.shape[0] - 1

    #define b0
    b0 = best_model.intercept_[0]

    print(f"n = {n}")
    print(f"b0 = {b0:.4f}")

    #adjust characteristic name in best_model_summary_table
    numeric_col = config_data['numeric_col']
    for col in best_model_summary['Characteristic']:

        if col in numeric_col:
            bin_col = col + '_binned'
        else:
            bin_col = col

        best_model_summary.replace(col, bin_col, inplace = True) 

    #merge tables to get beta/parameter estimate for each characteristic
    scorecards = pd.merge(left = WOE_table,
                          right = best_model_summary,
                          how = 'left',
                          on = ['Characteristic'])
    
    #define beta and WOE
    beta = scorecards['Estimate']
    WOE = scorecards['WOE']

    #calculate the score point for each attribute
    scorecards['Points'] = (offset/n) - factor*((b0/n) + (beta*WOE))
    scorecards['Points'] = scorecards['Points'].astype('int')

    #validate
    print('Scorecards table shape : ', scorecards.shape)
    
    #dump the scorecards
    scorecards_path = config_data['scorecards_path']
    utils.pickle_dump(scorecards, scorecards_path)

    return scorecards

#generate the Points map dict function
def get_points_map_dict():
    """
    Generate a Points mapping dictionary based on the calculated score points for each attribute.

    This function creates a mapping dictionary that assigns score points to each attribute based on the scorecards table.
    The dictionary is structured to handle both categorical and numerical attributes.
    The resulting mapping dictionary is saved as a pickle file.

    Returns
    -------
    dict: A mapping dictionary that assigns score points to attributes.
    """
    #load the Scorecards table
    scorecards = utils.pickle_load(config_data['scorecards_path'])

    #initialize the dictionary
    points_map_dict = {}
    points_map_dict['Missing'] = {}
    unique_char = set(scorecards['Characteristic'])
    for char in unique_char:
        #get the Attribute & WOE info for each characteristics
        current_data = (scorecards
                            [scorecards['Characteristic']==char]  
                            [['Attribute', 'Points']])  
        
        #get the mapping
        points_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            points = current_data.loc[idx, 'Points']

            if attribute == 'Missing':
                points_map_dict['Missing'][char] = points
            else:
                points_map_dict[char][attribute] = points
                points_map_dict['Missing'][char] = np.nan

    #validate data
    print('Number of key : ', len(points_map_dict.keys()))

    #dump
    utils.pickle_dump(points_map_dict, config_data['points_map_dict_path'])

    return points_map_dict

def transform_points(raw_data=None, type=None, config_data=None):
    """
    Replace data values with score points based on the mapping dictionary.

    Args
    ----
    raw_data (DataFrame): The raw data to be transformed.
    type (str): The type of data being transformed (e.g., 'train', 'test', 'validation').
    config_data (dict): Configuration data.

    Returns
    -------
    DataFrame: The transformed data with score points.
    """
    #lLoad the numerical columns
    num_cols = config_data['numeric_col']

    #load the points_map_dict
    points_map_dict = utils.pickle_load(config_data['points_map_dict_path'])

    #load the saved data if type is not None
    if type is not None:
        raw_data = utils.pickle_load(config_data[f'{type}_path'][0])

    #map the data
    points_data = raw_data.copy()
    for col in points_data.columns:
        if col in num_cols:
            map_col = col + '_binned'
        else:
            map_col = col    

        points_data[col] = points_data[col].map(points_map_dict[map_col])

    #map the data if there is a missing value or out of range value
    for col in points_data.columns:
        if col in num_cols:
            map_col = col + '_binned'
        else:
            map_col = col 

        points_data[col] = points_data[col].fillna(value=points_map_dict['Missing'][map_col])

    #dump data
    if type is not None:
        utils.pickle_dump(points_data, config_data[f'X_{type}_points_path'])

    return points_data   

#function to predict the credit score
def predict_score(raw_data, config_data):
    """
    Predict the credit score based on the transformed data.

    Args
    ----
    raw_data (DataFrame): The raw data for which the credit score is to be predicted.
    config_data (dict): Configuration data.

    Returns
    -------
    int: The predicted credit score.
    """
    
    points = transform_points(raw_data = raw_data, 
                              type = None, 
                              config_data = config_data)
    
    score = int(points.sum(axis=1))

    utils.pickle_dump(score, config_data['score_path'])

    return score

if __name__ == "__main__":

    #load config file
    config_data = utils.config_load()

    #create the scorecards
    scaling()

    #generate the points map dict
    get_points_map_dict()

    #predict the score
    transform_points(type='train', config_data=config_data)