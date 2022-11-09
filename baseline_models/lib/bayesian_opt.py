from types import SimpleNamespace
from .cross_val import cross_val, cross_val_fp   
from hyperopt import fmin, tpe
from functools import partial


def bayesian_opt(df, space, objective, model_class, n_train=0.8, max_eval=32):
    """
    Overarching function for Bayesian optimization

    Args:
        df (pd.DataFrame): dataframe containing the data points
        space (dict): dictionary containing the parameters for the selected regressor
        objective (function): specific objective function to be used
        model_class (Model): the abstract model class to initialize in every iteration
        n_train (float, optional): fraction of the training data to use. Defaults to 0.8.
        max_eval (int, optional): number of iterations to perform. Defaults to 32

    Returns:
        dict: optimal parameters for the selected regressor
    """
    df_sample = df.sample(frac=n_train)
    fmin_objective = partial(objective, data=df_sample, model_class=model_class)    
    best = fmin(fmin_objective, space, algo=tpe.suggest, max_evals=max_eval)

    return best


def objective_rf_fp(args_dict, data, model_class):
    """
    Objective function for random forest Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the RF regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(n_estimators=int(args.n_estimators), 
                                    max_features=args.max_features,
                                    min_samples_leaf=int(args.min_samples_leaf),
                                    random_state=2)
    cval,_ = cross_val_fp(data, estimator, 4)

    return cval.mean() 


def objective_rf(args_dict, data, model_class):
    """
    Objective function for random forest Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the RF regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(n_estimators=int(args.n_estimators), 
                                max_features=args.max_features,
                                min_samples_leaf=int(args.min_samples_leaf),
                                random_state=2)
    cval,_ = cross_val(data, estimator, 4)


    return cval.mean() 


def objective_xgboost(args_dict, data, model_class):
    """
    Objective function for xgboost Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the xgboost regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(max_depth=int(args.max_depth), 
                                    gamma=args.gamma,
                                    learning_rate=args.learning_rate,
                                    min_child_weight=args.min_child_weight,
                                    n_estimators=int(args.n_estimators))
    cval,_ = cross_val(data, estimator, 4)

    return cval.mean()     


def objective_xgboost_fp(args_dict, data, model_class):
    """
    Objective function for xgboost Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the RF regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(max_depth=int(args.max_depth), 
                                    gamma=args.gamma,
                                    learning_rate=args.learning_rate,
                                    min_child_weight=args.min_child_weight,
                                    n_estimators=int(args.n_estimators))
    cval,_ = cross_val_fp(data, estimator, 4)

    return cval.mean()


def objective_knn_fp(args_dict, data, model_class):
    """
    Objective function for knn Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the RF regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(n=int(args.n),
                    dipole_dist_weight=args.lam,
                    dipolarophile_dist_weight=(1-args.lam)*args.mu,
                    product_dist_weight=(1-args.lam)*(1-args.mu))

    cval,_ = cross_val_fp(data, estimator, 4, knn=True)

    return cval.mean() 


def objective_SVR(args_dict, data, model_class):
    """
    Objective function for SVR Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the SVR regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    kernel_dict = {1: 'linear', 2: 'rbf', 3: 'poly'}
    estimator = model_class(kernel=kernel_dict[int(args.kernel)], C=args.C, gamma=args.gamma, epsilon=args.epsilon)
    cval,_ = cross_val(data, estimator, 4)

    return cval.mean()
