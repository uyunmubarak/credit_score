#import library
import pandas as pd
import numpy as np
import utils as utils

#function to generate the WOE mapping dictionary
def get_woe_map_dict():
    """
    Generate a Weight of Evidence (WOE) mapping dictionary for transforming raw data into WOE values.

    Returns
    -------
    dict: A dictionary that maps characteristics, attributes, and their respective WOE values.
    """
    #load the WOE table
    WOE_table = utils.pickle_load(config_data['WOE_table_path'])

    #initialize the dictionary
    WOE_map_dict = {}
    WOE_map_dict['Missing'] = {}
    
    unique_char = set(WOE_table['Characteristic'])
    for char in unique_char:
        #get the Attribute & WOE info for each characteristics
        current_data = (WOE_table
                            [WOE_table['Characteristic']==char]     
                            [['Attribute', 'WOE']])                 
        
        #get the mapping
        WOE_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            woe = current_data.loc[idx, 'WOE']

            if attribute == 'Missing':
                WOE_map_dict['Missing'][char] = woe
            else:
                WOE_map_dict[char][attribute] = woe
                WOE_map_dict['Missing'][char] = np.nan

    #validate data
    print('Number of key : ', len(WOE_map_dict.keys()))

    #dump
    utils.pickle_dump(WOE_map_dict, config_data['WOE_map_dict_path'])

    return WOE_map_dict

#function to replace the raw data in the train set with WOE values
def transform_woe(raw_data=None, type=None, config_data=None):
    """
    Replace raw data values with corresponding Weight of Evidence (WOE) values based on the WOE mapping dictionary.

    Args
    ----
    raw_data (pd.DataFrame, optional): The raw data to be transformed. If not provided, the data is loaded from a file.
    type (str, optional): The type of data (e.g., 'train' or 'test').
    config_data (dict): The configuration data.

    Returns
    -------
    pd.DataFrame: The transformed data with WOE values.
    """
    #load the numerical columns
    numeric_col = config_data['numeric_col']

    #load the WOE_map_dict
    WOE_map_dict = utils.pickle_load(config_data['WOE_map_dict_path'])

    #load the saved data if type is not None
    if type is not None:
        raw_data = utils.pickle_load(config_data[f'{type}_path'][0])

    #map the data
    woe_data = raw_data.copy()
    for col in woe_data.columns:
        if col in numeric_col:
            map_col = col + '_binned'
        else:
            map_col = col    

        woe_data[col] = woe_data[col].map(WOE_map_dict[map_col])

    #map the data if there is a missing value or out of range value
    for col in woe_data.columns:
        if col in numeric_col:
            map_col = col + '_binned'
        else:
            map_col = col 

        woe_data[col] = woe_data[col].fillna(value=WOE_map_dict['Missing'][map_col])

    #validate
    print('Raw data shape : ', raw_data.shape)
    print('WOE data shape : ', woe_data.shape)

    #dump data
    if type is not None:
        utils.pickle_dump(woe_data, config_data[f'X_{type}_woe_path'])

    return woe_data

if __name__ == "__main__":
    #load config file
    config_data = utils.config_load()

    #generate the WOE map dict
    get_woe_map_dict()

    #transform the raw train set into WOE values
    transform_woe(type='train', config_data=config_data)