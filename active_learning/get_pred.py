import pandas as pd
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--iteration', type=int, required=True,
                    help='iteration of the active learning loop')
parser.add_argument('--predictions-file', type=str, required=True,
                    help='path to the predictions file.')
parser.add_argument('--rxn-smiles-file', type=str, required=True,
                    help='path to .csv file containing the reaction SMILES')


if __name__ == '__main__':
    args = parser.parse_args()
    df_pred_val = pd.read_csv(args.predictions_file)
    df_rxn_smiles = pd.read_csv(args.rxn_smiles_file)

    df_rxn_smiles.set_index('rxn_id', inplace=True)
    df_pred_val['rxn_smiles'] = df_pred_val['rxn_id'].apply(lambda x: df_rxn_smiles.loc[x]['rxn_smiles'])

    df_pred_val[['rxn_id', 'rxn_smiles', 'predicted_activation_energy', 'predicted_reaction_energy']].to_csv(
        f'predictions_{args.predictions_file.split("/")[-1].split("_")[0]}_iteration{args.iteration}.csv')
