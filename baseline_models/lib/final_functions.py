from .cross_val import cross_val, cross_val_fp
from sklearn.linear_model import LinearRegression
from hyperopt import hp
from .bayesian_opt import bayesian_opt, objective_rf, objective_rf_fp
from .bayesian_opt import objective_xgboost, objective_xgboost_fp
from .bayesian_opt import objective_SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from .bayesian_opt import objective_knn_fp
from .k_nearest_neighbors import KNN

n_estimator_dict = {0: 10, 1: 30, 2: 50, 3: 100, 4: 200, 5: 300, 6: 400, 7: 600}
min_samples_leaf_dict = {0: 1, 1: 2, 2: 5, 3: 10, 4: 20, 5: 50}

def get_cross_val_accuracy_linear_regression(df, logger, n_fold, split_dir=None):
    """
    Get the linear regression accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = LinearRegression()
    rmse, mae = cross_val(df, model, n_fold, split_dir=split_dir)

    logger.info(f'{n_fold}-fold CV RMSE and MAE for linear regression: {rmse} {mae}')


def get_optimal_parameters_xgboost_descriptors(df, logger, max_eval=32):
    """
    Get the optimal descriptors for xgboost (descriptors) through Bayesian optimization.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'max_depth': hp.quniform('max_depth', low=2, high=10, q=1),
        'gamma': hp.qloguniform('gamma', low=0.0, high=6.0, q=2.0),
        'n_estimators': hp.quniform('n_estimators', low=100, high=800, q=100),
        'learning_rate': hp.quniform('learning_rate', low=0.05, high=0.20, q=0.05),
        'min_child_weight': hp.quniform('min_child_weight', low=2, high=10, q=2.0)
    }
    optimal_parameters = bayesian_opt(df, space, objective_xgboost, XGBRegressor, max_eval=max_eval)
    logger.info(f'Optimal parameters for xgboost -- descriptors: {optimal_parameters}')

    return optimal_parameters


def get_cross_val_accuracy_xgboost_descriptors(df, logger, n_fold, parameters, split_dir=None):
    """
    Get the xgboost (descriptors) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = XGBRegressor(max_depth=int(parameters['max_depth']), 
                        gamma=parameters['gamma'], 
                        n_estimators=int(parameters['n_estimators']),
                        learning_rate=parameters['learning_rate'],
                        min_child_weight=parameters['min_child_weight'])
    rmse, mae = cross_val(df, model, n_fold, split_dir=split_dir)
    logger.info(f'{n_fold}-fold CV RMSE and MAE for xgboost -- descriptors: {rmse} {mae}')
    logger.info(f'Parameters used: {parameters}')


def get_optimal_parameters_xgboost_fp(df_fp, logger, max_eval=32):
    """
    Get the optimal descriptors for xgboost (fingerprints) through Bayesian optimization.

    Args:
        df_fp (pd.DataFrame): input dataframe containing the fingerprints
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'max_depth': hp.quniform('max_depth', low=2, high=10, q=1),
        'gamma': hp.qloguniform('gamma', low=0.0, high=6.0, q=2.0),
        'n_estimators': hp.quniform('n_estimators', low=100, high=800, q=100),
        'learning_rate': hp.quniform('learning_rate', low=0.05, high=0.20, q=0.05),
        'min_child_weight': hp.quniform('min_child_weight', low=2, high=10, q=2.0)
    }
    optimal_parameters = bayesian_opt(df_fp, space, objective_xgboost_fp, XGBRegressor, max_eval=max_eval)
    logger.info(f'Optimal parameters for xgboost -- fingerprints: {optimal_parameters}')

    return optimal_parameters


def get_cross_val_accuracy_xgboost_fp(df_fp, logger, n_fold, parameters, split_dir=None):
    """
    Get the xgboost (fingerprints) accuracy in cross-validation.

    Args:
        df_fp (pd.DataFrame): input dataframe containing the fingerprints
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = XGBRegressor(max_depth=int(parameters['max_depth']), 
                        gamma=parameters['gamma'], 
                        n_estimators=int(parameters['n_estimators']),
                        learning_rate=parameters['learning_rate'],
                        min_child_weight=parameters['min_child_weight'])
    rmse, mae = cross_val_fp(df_fp, model, n_fold, split_dir=split_dir)
    logger.info(f'{n_fold}-fold CV RMSE and MAE for xgboost -- fingerprints: {rmse} {mae}')
    logger.info(f'Parameters used: {parameters}')


def get_optimal_parameters_svr_descriptors(df, logger, max_eval=32):
    """
    Get the optimal descriptors for SVR (descriptors) through Bayesian optimization.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'kernel': hp.quniform('kernel', low=1, high=3, q=1),
        'C': hp.loguniform('C', low=0, high=100, q=0.1),
        'gamma': hp.loguniform('gamma', low=1e-8, high=1e-2),
        'epsilon': hp.loguniform('epsilon', low=0, high=1)
    }
    optimal_parameters = bayesian_opt(df, space, objective_SVR, SVR, max_eval=max_eval)
    logger.info(f'Optimal parameters for SVR -- descriptors: {optimal_parameters}')

    return optimal_parameters


def get_cross_val_accuracy_svr_descriptors(df, logger, n_fold, parameters, split_dir=None):
    """
    Get the SVR (descriptors) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    kernel_dict = {1: 'linear', 2: 'rbf', 3: 'poly'}
    model = SVR(kernel=kernel_dict[int(parameters['kernel'])], C=parameters['C'], 
                gamma=parameters['gamma'], epsilon=parameters['epsilon'])
    rmse, mae = cross_val(df, model, n_fold, split_dir=split_dir)
    logger.info(f'{n_fold}-fold CV RMSE and MAE for SVR -- descriptors: {rmse} {mae}')
    logger.info(f'Parameters used: {parameters}')


def get_optimal_parameters_rf_descriptors(df, logger, max_eval=32):
    """
    Get the optimal descriptors for random forest (descriptors) through Bayesian optimization.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'n_estimators': hp.choice('n_estimators', [10, 30, 50, 100, 150, 200, 300, 400, 600]),
        'max_features': hp.quniform('max_features', low=0.1, high=1, q=0.1),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 5, 10, 20, 50])
    }

    optimal_parameters = bayesian_opt(df, space, objective_rf, RandomForestRegressor, max_eval=max_eval)
    optimal_parameters['n_estimators'] = n_estimator_dict[optimal_parameters['n_estimators']]
    optimal_parameters['min_samples_leaf'] = min_samples_leaf_dict[optimal_parameters['min_samples_leaf']]
    logger.info(f'Optimal parameters for RF -- descriptors: {optimal_parameters}')

    return optimal_parameters


def get_cross_val_accuracy_rf_descriptors(df, logger, n_fold, parameters, split_dir=None):
    """
    Get the random forest (descriptors) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = RandomForestRegressor(n_estimators=int(parameters['n_estimators']), 
            max_features=parameters['max_features'], min_samples_leaf=int(parameters['min_samples_leaf']))
    rmse, mae = cross_val(df, model, n_fold, split_dir=split_dir)
    logger.info(f'{n_fold}-fold CV RMSE and MAE for RF -- descriptors: {rmse} {mae}')
    logger.info(f'Parameters used: {parameters}')


def get_optimal_parameters_rf_fp(df_fp, logger, max_eval=32):
    """
    Get the optimal descriptors for random forest (fingerprints) through Bayesian optimization.

    Args:
        df_fp (pd.DataFrame): input dataframe containing the fingerprints
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'n_estimators': hp.choice('n_estimators', [10, 30, 50, 100, 150, 200, 300, 400, 600]),
        'max_features': hp.quniform('max_features', low=0.1, high=1, q=0.1),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 5, 10, 20, 50])
    }
    optimal_parameters = bayesian_opt(df_fp, space, objective_rf_fp, RandomForestRegressor, max_eval=max_eval)
    optimal_parameters['n_estimators'] = n_estimator_dict[optimal_parameters['n_estimators']]
    optimal_parameters['min_samples_leaf'] = min_samples_leaf_dict[optimal_parameters['min_samples_leaf']]
    logger.info(f'Optimal parameters for RF -- fingerprints: {optimal_parameters}')

    return optimal_parameters


def get_cross_val_accuracy_rf_fp(df_fp, logger, n_fold, parameters, split_dir=None):
    """
    Get the random forest (descriptors) accuracy in cross-validation.

    Args:
        df_fp (pd.DataFrame): input dataframe containing the fingerprints
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = RandomForestRegressor(n_estimators=int(parameters['n_estimators']), 
                                max_features=parameters['max_features'], 
                                min_samples_leaf=int(parameters['min_samples_leaf']))
    rmse, mae = cross_val_fp(df_fp, model, n_fold, split_dir=split_dir)
    logger.info(f'{n_fold}-fold CV RMSE and MAE for RF -- fingerprints: {rmse} {mae}')
    logger.info(f'Parameters used: {parameters}')


def get_optimal_parameters_knn_fp(df_fp, logger, max_eval=32):
    """
    Get the optimal descriptors for KNN (fingerprints) through Bayesian optimization.

    Args:
        df_fp (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'n': hp.quniform('n', low=3, high=7, q=2),
        'lam': hp.quniform('lam', low=0, high=1, q=0.1),
        'mu': hp.quniform('mu', low=0, high=1, q=0.1)
    }
    optimal_parameters = bayesian_opt(df_fp, space, objective_knn_fp, KNN, max_eval=max_eval)
    logger.info(f'Optimal parameters for KNN -- fingerprints: {optimal_parameters}')

    return optimal_parameters


def get_cross_val_accuracy_knn_fp(df_fp, logger, n_fold, parameters, split_dir=None):
    """
    Get the KNN (descriptors) accuracy in cross-validation.

    Args:
        df_fp (pd.DataFrame): input dataframe containing the fingerprints
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = KNN(n=int(parameters['n']), 
                dipole_dist_weight=parameters['lam'], dipolarophile_dist_weight=(1 - parameters['lam']) * parameters['mu'],
                product_dist_weight=(1 - parameters['lam']) * (1 - parameters['mu'])
            )
    rmse, mae = cross_val_fp(df_fp, model, n_fold, knn=True, split_dir=split_dir)
    logger.info(f'{n_fold}-fold CV RMSE and MAE for k-nearest neighbors (fingerprints): {rmse} {mae}')
    logger.info(f'Parameters used: {parameters}')
