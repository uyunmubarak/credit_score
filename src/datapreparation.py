#import library
import pandas as pd
import utils as utils
from sklearn.model_selection import train_test_split

def read_data():
    """
    Load data from a CSV file, select relevant columns, and save it using pickle.

    Returns
    -------
    pd.DataFrame: Dataframe containing selected columns.
    """

    #load data
    data_path = config_data['raw_dataset_path']
    data = pd.read_csv(data_path)

    #validate data shape
    print("Data shape       :", data.shape)

    data = data[['Age', 'Income', 'LoanAmount', 'MonthsEmployed', 'NumCreditLines', 'InterestRate',
                       'LoanTerm', 'DTIRatio', 'Education', 'EmploymentType', 'MaritalStatus',
                       'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner', 'Default']]

    #pickle dumping (save the result)
    dump_path = config_data['dataset_path']
    utils.pickle_dump(data, dump_path)

    return data

def splitting_data():
    """
    Split input (predictors) and output (responses), and save them as pickle files.

    Returns
    -------
    pd.DataFrame: Predictors (X) and responses (y).
    """

    dataset_path = config_data['dataset_path']
    data = utils.pickle_load(dataset_path)

    #define y
    response_variable = config_data['response_variable']
    y = data[response_variable]

    #define X
    X = data.drop(columns = [response_variable],
                  axis = 1)
    
    #validate the splitting
    print('y shape :', y.shape)
    print('X shape :', X.shape)

    #dumping
    dump_path_predictors = config_data['predictors_set_path']
    utils.pickle_dump(X, dump_path_predictors)

    dump_path_response = config_data['response_set_path']    
    utils.pickle_dump(y, dump_path_response)
    
    return X, y

def split_train_test():
    """
    Split the data into training and testing sets and save them as pickle files.

    Returns
    -------
    pd.DataFrame: Training predictors (X_train), testing predictors (X_test), 
    training responses (y_train), and testing responses (y_test).
    """
    
    #load the X and y
    X = utils.pickle_load(config_data['predictors_set_path'])
    y = utils.pickle_load(config_data['response_set_path'])

    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify = y,
                                                        test_size = config_data['test_size'],
                                                        random_state = 42)
    #validate splitting
    print('X_train shape :', X_train.shape)
    print('y_train shape :', y_train.shape)
    print('X_test shape  :', X_test.shape)
    print('y_test shape  :', y_test.shape)

    #dump data
    utils.pickle_dump(X_train, config_data['train_path'][0])
    utils.pickle_dump(y_train, config_data['train_path'][1])
    utils.pickle_dump(X_test, config_data['test_path'][0])
    utils.pickle_dump(y_test, config_data['test_path'][1])

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    #load config data
    config_data = utils.config_load()

    #run all functions
    read_data()
    splitting_data()
    split_train_test()