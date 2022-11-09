import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os


def cross_val(df, model, n_folds, sample=None, split_dir=None):
    """
    Function to perform cross-validation

    Args:
        df (pd.DataFrame): the DataFrame containing features and targets
        model (sklearn.Regressor): An initialized sklearn model
        n_folds (int): the number of folds
        sample(int): the size of the subsample for the training set (default = None)
        split_dir (str): the path to a directory containing data splits. If None, random splitting is performed.

    Returns:
        int: the obtained RMSE
    """
    rmse_list, mae_list = [], []
    feature_names = [column for column in df.columns if column not in['rxn_id', 'DG_TS', 'G_r', 'train']]

    if split_dir == None:
        df = df.sample(frac=1, random_state=0)
        chunk_list = np.array_split(df, n_folds)

    for i in range(n_folds):
        if split_dir == None:
            df_train = pd.concat([chunk_list[j] for j in range(n_folds) if j != i])
            if sample != None:
                df_train = df_train.sample(n=sample)
            df_test = chunk_list[i]
        else:
            rxn_ids_train1 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/train.csv'))[['rxn_id']].values.tolist()
            rxn_ids_train2 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/valid.csv'))[['rxn_id']].values.tolist()
            rxn_ids_train = list(np.array(rxn_ids_train1 + rxn_ids_train2).reshape(-1))
            df['train'] = df['rxn_id'].apply(lambda x: int(x) in rxn_ids_train)
            df_train = df[df['train'] == True]
            df_test = df[df['train'] == False]

        X_train, y_train = df_train[feature_names], df_train[['DG_TS']]
        X_test, y_test = df_test[feature_names], df_test[['DG_TS']]

        # scale the two dataframes
        feature_scaler = StandardScaler()
        feature_scaler.fit(X_train)
        X_train = feature_scaler.transform(X_train)
        X_test = feature_scaler.transform(X_test)

        target_scaler = StandardScaler()
        target_scaler.fit(y_train)
        y_train = target_scaler.transform(y_train)
        y_test = target_scaler.transform(y_test)

        # fit and compute rmse and mae
        model.fit(X_train, y_train.ravel())
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1,1)

        rmse_fold = np.sqrt(mean_squared_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test)))
        rmse_list.append(rmse_fold)

        mae_fold = mean_absolute_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test))
        mae_list.append(mae_fold)

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))

    return rmse, mae

def cross_val_fp(df_fp, model, n_folds, knn=False, split_dir=None):
    """
    Function to perform cross-validation with fingerprints

    Args:
        df_fp (pd.DataFrame): the DataFrame containing fingerprints and targets
        model (sklearn.Regressor): An initialized sklearn model
        n_folds (int): the number of folds
        knn (bool): whether the model is a KNN model
        split_dir (str): the path to a directory containing data splits. If None, random splitting is performed.

    Returns:
        int: the obtained RMSE
    """
    rmse_list, mae_list = [], []

    if split_dir == None:
        df_fp = df_fp.sample(frac=1, random_state=0)
        chunk_list = np.array_split(df_fp, n_folds)

    for i in range(n_folds):
        if split_dir == None:
            df_train = pd.concat([chunk_list[j] for j in range(n_folds) if j != i])
            df_test = chunk_list[i]
        else:
            rxn_ids_train1 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/train.csv'))[['rxn_id']].values.tolist()
            rxn_ids_train2 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/valid.csv'))[['rxn_id']].values.tolist()
            rxn_ids_train = list(np.array(rxn_ids_train1 + rxn_ids_train2).reshape(-1))
            df_fp['train'] = df_fp['rxn_id'].apply(lambda x: int(x) in rxn_ids_train)
            df_train = df_fp[df_fp['train'] == True]
            df_test = df_fp[df_fp['train'] == False]

        y_train = df_train[['DG_TS']]
        y_test = df_test[['DG_TS']]

        # scale targets
        target_scaler = StandardScaler()
        target_scaler.fit(y_train)
        y_train = target_scaler.transform(y_train)
        y_test = target_scaler.transform(y_test) 

        if knn:
            X_train = [np.vstack(df_train['fingerprint_dipole'].to_numpy()), 
                    np.vstack(df_train['fingerprint_dipolarophile'].to_numpy()), 
                    np.vstack(df_train['fingerprint_product'].to_numpy())]
            X_test = [np.vstack(df_test['fingerprint_dipole'].to_numpy()), 
                    np.vstack(df_test['fingerprint_dipolarophile'].to_numpy()), 
                    np.vstack(df_test['fingerprint_product'].to_numpy())]
        else:
            X_train = []
            for fp in df_train['fingerprint'].values.tolist():
                X_train.append(list(fp))
            X_test = []
            for fp in df_test['fingerprint'].values.tolist():
                X_test.append(list(fp))

        # fit and compute rmse
        model.fit(X_train, y_train.ravel())
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1,1)

        rmse_fold = np.sqrt(mean_squared_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test)))
        rmse_list.append(rmse_fold)

        mae_fold = mean_absolute_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test))
        mae_list.append(mae_fold)

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))

    return rmse, mae 
