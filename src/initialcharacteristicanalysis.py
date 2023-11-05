#import library
import utils as utils
import pandas as pd
import numpy as np

def concat_data(type):
    """
    Concatenate input (X) and output (y) data and save it as a pickle file.

    Args
    ----
    type (str): The type of data (e.g., 'train' or 'test').

    Returns
    -------
    pd.DataFrame: The concatenated data.
    """
    X = utils.pickle_load(config_data[f'{type}_path'][0])
    y = utils.pickle_load(config_data[f'{type}_path'][1])
    
    #concatenate X and y
    data = pd.concat((X, y),
                     axis = 1)

    #validate data
    print(f'Data shape:', data.shape)

    #dump concatenated data
    utils.pickle_dump(data, config_data[f'data_{type}_path'])
   
    return data

#create a function for binning the numerical predictor
def create_binning(data, predictor_label, num_of_bins):
    """
    Bin the numerical predictor and add the binned column to the data.

    Args
    ----
    data (pd.DataFrame): The data containing the numerical predictor.
    predictor_label (str): The label of the numerical predictor to be binned.
    num_of_bins (int): The number of bins to use for binning.

    Returns
    -------
    pd.DataFrame: The data with the binned column.
    """
    #create a new column containing the binned predictor
    data[predictor_label + "_binned"] = pd.qcut(data[predictor_label],
                                             q = num_of_bins,
                                             duplicates='drop')

    return data

def binned_data(type):
    """
    Bin the numerical and missing data, and save it as a pickle file.

    Args
    ----
    type (str): The type of data (e.g., 'train' or 'test').

    Returns
    -------
    pd.DataFrame: The binned data.
    """
    #load the concatenated data
    data = utils.pickle_load(config_data[f'data_{type}_path'])

    #bin the numerical columns
    numeric_col = config_data['numeric_col']
    num_of_bins = config_data['num_of_bins']

    for column in numeric_col:
        bin_data = create_binning(data = data,
                                         predictor_label = column,
                                         num_of_bins = num_of_bins)

    #validate
    print(f"Original data shape : ", data.shape)
    print(f"Binned data shape  : ", bin_data.shape)

    #dump binned data
    utils.pickle_dump(bin_data, config_data[f'data_{type}_binned_path'])
        
    return bin_data

def create_list_crosstab():
    """
    Generate the crosstab list (contingency table) for WOE and IV calculation. Only in training data.

    Returns
    -------
    list: List of contingency tables.
    """
    #load the binned train data
    bin_training_data = utils.pickle_load(config_data['data_train_binned_path'])

    #load the response variable (we will summarize based on the response variable)
    response_variable = config_data['response_variable']

    #iterate over numercial columns
    numeric_crosstab = []
    numeric_col = config_data['numeric_col']
    for column in numeric_col:
        #create a contingency table
        crosstab = pd.crosstab(bin_training_data[column + "_binned"],
                               bin_training_data[response_variable],
                               margins = True)

        #append to the list
        numeric_crosstab.append(crosstab)

    #iterate over categorical columns
    categoric_crosstab = []
    categoric_col = config_data['categoric_col']
    for column in categoric_col:
        #create a contingency table
        crosstab = pd.crosstab(bin_training_data[column],
                               bin_training_data[response_variable],
                               margins = True)

        #append to the list
        categoric_crosstab.append(crosstab)

    #put all two in a crosstab_list
    list_crosstab = numeric_crosstab + categoric_crosstab

    #validate the crosstab_list
    print('number of num bin : ', [bin.shape for bin in numeric_crosstab])
    print('number of cat bin : ', [bin.shape for bin in categoric_crosstab])

    #dump the result
    utils.pickle_dump(list_crosstab, config_data['crosstab_list_path'])

    return list_crosstab

def WOE_and_IV():
    """
    Calculate the WoE (Weight of Evidence) and IV (Information Value) for each characteristic.

    Returns
    -------
    pd.DataFrame: WoE table and IV table.
    """
    #load the crosstab list
    list_crosstab = utils.pickle_load(config_data['crosstab_list_path'])

    #create initial storage for WoE and IV
    WOE_list, IV_list = [], []
    
    #perform the calculation for all crosstab list
    for crosstab in list_crosstab:
        #calculate the WoE and IV
        #calculate % Good
        crosstab['p_good'] = crosstab[0]/crosstab[0]['All']    
        #calculate % Bad                             
        crosstab['p_bad'] = crosstab[1]/crosstab[1]['All']  
        #calculate the WOE                               
        crosstab['WOE'] = np.log(crosstab['p_good']/crosstab['p_bad']) 
        #calculate the contribution value for IV                    
        crosstab['contribution'] = (crosstab['p_good']-crosstab['p_bad'])*crosstab['WOE']   
                
        #append to list
        IV = crosstab['contribution'][:-1].sum()
        add_IV = {'Characteristic': crosstab.index.name, 
                  'Information Value': IV}

        WOE_list.append(crosstab)
        IV_list.append(add_IV)


    #create WOE table
    #create initial table to summarize the WOE values
    WOE_table = pd.DataFrame({'Characteristic': [],
                              'Attribute': [],
                              'WOE': []})
    for i in range(len(list_crosstab)):
        #define crosstab and reset index
        crosstab = list_crosstab[i].reset_index()

        #save the characteristic name
        char_name = crosstab.columns[0]

        #only use two columns (Attribute name and its WOE value)
        #drop the last row (average/total WOE)
        crosstab = crosstab.iloc[:-1, [0,-2]]
        crosstab.columns = ['Attribute', 'WOE']

        #add the characteristic name in a column
        crosstab['Characteristic'] = char_name

        WOE_table = pd.concat((WOE_table, crosstab), 
                                axis = 0)

        #reorder the column
        WOE_table.columns = ['Characteristic',
                            'Attribute',
                            'WOE']
    

    #create IV table
    #create the initial table for IV
    IV_table = pd.DataFrame({'Characteristic': [],
                             'Information Value' : []})
    IV_table = pd.DataFrame(IV_list)

    #define the predictive power of each characteristic
    strength = []

    #assign the rule of thumb regarding IV
    for iv in IV_table['Information Value']:
        if iv < 0.02:
            strength.append('Unpredictive')
        elif iv >= 0.02 and iv < 0.1:
            strength.append('Weak')
        elif iv >= 0.1 and iv < 0.3:
            strength.append('Medium')
        else:
            strength.append('Strong')

    #assign the strength to each characteristic
    IV_table = IV_table.assign(Strength = strength)

    #sort the table by the IV values
    IV_table = IV_table.sort_values(by='Information Value')
    
    #validate
    print('WOE table shape : ', WOE_table.shape)
    print('IV table shape  : ', IV_table.shape)

    #dump data
    utils.pickle_dump(WOE_table, config_data['WOE_table_path'])
    utils.pickle_dump(IV_table, config_data['IV_table_path']) 

    return WOE_table, IV_table

if __name__ == "__main__":
    #load config file
    config_data = utils.config_load()

    #concat and binning the train set
    concat_data(type='train')
    binned_data(type='train')

    #obtain the WoE and IV    
    create_list_crosstab()
    WOE_and_IV()