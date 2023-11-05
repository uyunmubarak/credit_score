#import library
import utils as utils
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

#function to perform forward selection procedure
def forward(X, y, predictors, scoring='roc_auc', cv=5):
    """
    Perform forward selection procedure to select the best predictors for a logistic regression model.

    Args
    ----
    X (numpy.ndarray): The feature matrix.
    y (numpy.ndarray): The target variable.
    predictors (list): List of predictor indices to be included in the model.
    scoring (str, optional): The scoring metric for cross-validation. Default is 'roc_auc'.
    cv (int, optional): Number of cross-validation folds. Default is 5.

    Returns
    -------
    pd.DataFrame: A table containing the CV scores and the selected predictors for each step of the forward selection.
    pd.Series: The best model with selected predictors.
    """

    #define sample size and  number of all predictors
    n_samples, n_predictors = X.shape

    #define list of all predictors
    col_list = np.arange(n_predictors)

    #define remaining predictors for each k
    remaining_predictors = [p for p in col_list if p not in predictors]

    #initialize list of predictors and its CV Score
    pred_list = []
    score_list = []

    #cross validate each possible combination of remaining predictors
    for p in remaining_predictors:
        combi = predictors + [p]

        #extract predictors combination
        X_ = X[:, combi]
        y_ = y

        #define the estimator
        model = LogisticRegression(penalty = None,
                                   class_weight = 'balanced')

        #cross validate the recall scores of the model
        cv_results = cross_validate(estimator = model,
                                    X = X_,
                                    y = y_,
                                    scoring = scoring,
                                    cv = cv)

        #calculate the average CV/recall score
        score_ = np.mean(cv_results['test_score'])

        #append predictors combination and its CV Score to the list
        pred_list.append(list(combi))
        score_list.append(score_)

    #tabulate the results
    models = pd.DataFrame({"Predictors": pred_list,
                           "CV Score": score_list})

    #choose the best model
    best_model = models.loc[models['CV Score'].argmax()]

    return models, best_model

#function to perform forward selection on all characteristics
def run_forward():
    """
    Perform forward selection procedure to select the best predictors for a logistic regression model.

    Args
    ----
    X (numpy.ndarray): The feature matrix.
    y (numpy.ndarray): The target variable.
    predictors (list): List of predictor indices to be included in the model.
    scoring (str, optional): The scoring metric for cross-validation. Default is 'roc_auc'.
    cv (int, optional): Number of cross-validation folds. Default is 5.

    Returns
    -------
    pd.DataFrame: A table containing the CV scores and the selected predictors for each step of the forward selection.
    pd.Series: The best model with selected predictors.
    """

    cv = config_data['num_of_cv']
    scoring = config_data['scoring']

    X_train_woe_path = config_data['X_train_woe_path']
    X_train_woe = utils.pickle_load(X_train_woe_path)
    X_train = X_train_woe.to_numpy()

    y_train_path = config_data['train_path'][1]
    y_train = utils.pickle_load(y_train_path)
    y_train = y_train.to_numpy()

    #define predictor for the null model
    predictor = []

    #the predictor in the null model is zero values for all predictors
    X_null = np.zeros((X_train.shape[0], 1))

    #define the estimator
    model = LogisticRegression(penalty = None,
                               class_weight = 'balanced')

    #cross validate
    cv_results = cross_validate(estimator = model,
                                X = X_null,
                                y = y_train,
                                cv = cv,
                                scoring = scoring)

    #calculate the average CV score
    score_ = np.mean(cv_results['test_score'])

    #create table for the best model of each k predictors
    #append the results of null model
    forward_models = pd.DataFrame({"Predictors": [predictor],
                                   "CV Score": [score_]})

    #define list of predictors
    predictors = []
    n_predictors = X_train.shape[1]

    #perform forward selection procedure for k=1,...,n_predictors
    for k in range(n_predictors):
        _, best_model = forward(X = X_train,
                                y = y_train,
                                predictors = predictors,
                                scoring = scoring,
                                cv = cv)

        #tabulate the best model of each k predictors
        forward_models.loc[k+1] = best_model
        predictors = best_model['Predictors']

    #find the best CV score
    best_idx = forward_models['CV Score'].argmax()
    best_cv_score = forward_models['CV Score'].loc[best_idx]
    best_predictors = forward_models['Predictors'].loc[best_idx]

    #print the summary
    print('===================================================')
    print('Best index            :', best_idx)
    print('Best CV Score         :', best_cv_score)
    print('Best predictors (idx) :', best_predictors)
    print('Best predictors       :')
    print(X_train_woe.columns[best_predictors].tolist())
    print('===================================================')

    print(forward_models)
    print('===================================================')
    
    forward_models_path = config_data['forward_models_path']
    utils.pickle_dump(forward_models, forward_models_path)

    best_predictors_path = config_data['best_predictors_path']
    utils.pickle_dump(best_predictors, best_predictors_path)

    return forward_models, best_predictors

#function to fit the best model on whole X_train
def best_model_fitting(best_predictors):
    """
    Fit the best model with the selected predictors on the entire training dataset and save the model and summary.

    Args
    ----
    best_predictors (list): List of predictor indices selected for the best model.

    Returns
    -------
    LogisticRegression: The trained logistic regression model with the selected predictors.
    pd.DataFrame: A summary of the model with coefficient estimates for each selected predictor.
    """

    X_train_path = config_data['X_train_woe_path']
    X_train_woe = utils.pickle_load(X_train_path)
    X_train = X_train_woe.to_numpy()

    y_train_path = config_data['train_path'][1]
    y_train = utils.pickle_load(y_train_path)
    y_train = y_train.to_numpy()

    if best_predictors is None:
        best_predictors_path = config_data['best_predictors_path']
        best_predictors = utils.pickle_load(best_predictors_path)
        print(f"Best predictors index   :", best_predictors)
    else:
        print(f"[Adjusted] best predictors index   :", best_predictors)

    #define X with best predictors
    X_train_best = X_train[:, best_predictors]

    #fit best model
    best_model = LogisticRegression(penalty = None,
                                    class_weight = 'balanced')
    best_model.fit(X_train_best, y_train)

    print(best_model)

    #extract the best model' parameter estimates
    best_model_intercept = pd.DataFrame({'Characteristic': 'Intercept',
                                         'Estimate': best_model.intercept_})
    
    best_model_params = X_train_woe.columns[best_predictors].tolist()

    best_model_coefs = pd.DataFrame({'Characteristic': best_model_params,
                                     'Estimate': np.reshape(best_model.coef_, 
                                                            len(best_predictors))})

    best_model_summary = pd.concat((best_model_intercept, best_model_coefs),
                                   axis = 0,
                                   ignore_index = True)
    
    print('===================================================')
    print(best_model_summary)
    
    best_model_path = config_data['best_model_path']
    utils.pickle_dump(best_model, best_model_path)

    best_model_summary_path = config_data['best_model_summary_path']
    utils.pickle_dump(best_model_summary, best_model_summary_path)

    return best_model, best_model_summary

if __name__ == "__main__":

    #load config file
    config_data = utils.config_load()

    #perform model selection on all characteristics
    run_forward()

    #fit the best model (with adjustment)
    best_model_fitting(best_predictors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])