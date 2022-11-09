import pandas as pd
from lib import get_cross_val_accuracy_linear_regression
from lib import get_optimal_parameters_xgboost_descriptors, get_cross_val_accuracy_xgboost_descriptors
from lib import get_optimal_parameters_xgboost_fp, get_cross_val_accuracy_xgboost_fp
from lib import get_optimal_parameters_rf_fp, get_cross_val_accuracy_rf_fp
from lib import get_optimal_parameters_rf_descriptors, get_cross_val_accuracy_rf_descriptors
from lib import get_optimal_parameters_knn_fp, get_cross_val_accuracy_knn_fp
from lib import create_logger
from lib import get_df_fingerprints, get_df_fingerprints_rp
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--csv-file', type=str, default='data_files/final_data.csv',
                    help='path to file containing the rxn-smiles')
parser.add_argument('--input-file', type=str, default='data_files/input_alt_models.pkl',
                    help='path to the input file')
parser.add_argument('--split_dir', type=str, default='data_files/splits',
                    help='path to the folder containing the requested splits for the cross validation')
parser.add_argument('--n-fold', type=int, default=10,
                    help='the number of folds to use during cross validation')       


if __name__ == '__main__':
    # set up
    args = parser.parse_args()
    logger = create_logger(args.input_file.split('/')[-1].split('_')[0])
    df = pd.read_pickle(args.input_file)
    df_rxn_smiles = pd.read_csv(args.csv_file)
    n_fold = args.n_fold
    split_dir = args.split_dir

    df_fp = get_df_fingerprints(df_rxn_smiles,2,2048)
    df_fp_knn = get_df_fingerprints_rp(df_rxn_smiles,2,2048)

    # linear regression
    get_cross_val_accuracy_linear_regression(df, logger, n_fold, split_dir)

    # KNN - fingerprints
    optimal_parameters_knn = get_optimal_parameters_knn_fp(df_fp_knn, logger, max_eval=64)
    get_cross_val_accuracy_knn_fp(df_fp_knn, logger, 10, optimal_parameters_knn, split_dir)

    # random forest - descriptors
    optimal_parameters_rf_descs = get_optimal_parameters_rf_descriptors(df, logger, max_eval=64)
    get_cross_val_accuracy_rf_descriptors(df, logger, 10, optimal_parameters_rf_descs, split_dir)  

    # random_forest - fingerprints
    optimal_parameters_rf_fp = get_optimal_parameters_rf_fp(df_fp, logger, max_eval=64)
    get_cross_val_accuracy_rf_fp(df_fp, logger, 10, optimal_parameters_rf_fp, split_dir)   

    # xgboost - descriptors
    optimal_parameters_xgboost_descs = get_optimal_parameters_xgboost_descriptors(df, logger, max_eval=128)
    get_cross_val_accuracy_xgboost_descriptors(df, logger, 10, optimal_parameters_xgboost_descs, split_dir)  

    # xgboost - fingerprints
    optimal_parameters_xgboost_fp = get_optimal_parameters_xgboost_fp(df_fp, logger, max_eval=128)
    get_cross_val_accuracy_xgboost_fp(df_fp, logger, 10, optimal_parameters_xgboost_fp, split_dir) 
