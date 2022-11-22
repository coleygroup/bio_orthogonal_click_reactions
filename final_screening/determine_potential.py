import pandas as pd
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--predictions-file', type=str, required=True,
                    help='input .csv file containing predicted reaction and activation energies')
parser.add_argument('--dipole-statistics-file', type=str, required=True,
                    help='.csv file containing the statistics for the individual dipoles')


def get_dipole_dict(statistics_file):
    df_dipoles = pd.read_csv(statistics_file)
    dipole_smiles = df_dipoles.dipole_smiles.tolist()
    G_act_min = df_dipoles.G_act_min.tolist()
    dipole_dict = dict(zip(dipole_smiles,G_act_min))
    
    return dipole_dict

def test(x):
    try:
        print(x['dipole'], dipole_dict[x['dipole']])
        return x['predicted_activation_energy'] - dipole_dict[x['dipole']]
    except:
        return 'None'


if __name__ == '__main__':
    args = parser.parse_args()
    df_pred = pd.read_csv(args.predictions_file)
    dipole_dict = get_dipole_dict(args.dipole_statistics_file)

    df_pred['bio_orthogonal_potential'] = df_pred.apply(
        lambda x: test(x), axis=1)

    print(len(df_pred), len(df_pred[df_pred['bio_orthogonal_potential'] == 'None']))
    print(df_pred[df_pred['bio_orthogonal_potential'] == 'None'].dipole.unique())
    df_pred.to_csv('test.csv')
    
