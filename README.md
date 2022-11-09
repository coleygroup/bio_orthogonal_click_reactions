# Dipolar_cycloaddition_dataset
This repository contains the code and auxiliary data associated to the "data-driven discovery of new bio-orthogonal click reactions" project. Code is provided "as-is". Minor edits may be required to tailor the scripts for different computational systems. Additionally, for some steps of the workflow, intermediate files need to be generated as a preliminary step (vide infra).

## Requirements

1. python 3.10
2. autode 1.2
3. rdkit 2020.03
4. ipykernel 6.9
5. pandas 1.4
6. pebble 4.6
7. xtb 6.3
8. tqdm 4.64
9. pip 2.22
10. rdchiral 1.1

Additionally, in order to execute the autodE high-throughput reaction profile computation workflow, Gaussian09/Gaussian16 needs to be accessible. More information about the autodE module can be found [here](https://github.com/duartegroup/autodE).

### Conda environment
To set up the conda environment:
```
conda env create --name <env-name> --file environment.yml
```

## Overview of the workflow

The complete workflow associated with this project can be broken down in the following elementary steps:

1. Define the search space of dipoles and dipolarophiles (both synthetic and biofragment-based examples), generate a representative dataset of cycloaddition reactions and compute the associated reaction profiles in a high-throughput manner.
2. Compute QM descriptors for each dipole and dipolarophile in a high-throughput manner.
3. Select an appropriate machine learning model architecture.
4. Generate an exhaustive list of reaction SMILES based on all dipole - biofragment-based dipolarophile combinations; generate the QM descriptor input for each generated reaction.
5. Iterate through an active learning loop to refine the dataset. This loop consists of the following steps:
      - Train an ML model on the current instance of the dataset.
      - Use the trained model to predict activation and reaction energies for all the bio-fragment, i.e., native, reactions.
      - Select promising dipoles for bio-orthogonal click applications based on the predictions made, i.e., retain only the dipoles which are not too reactive with the native dipolarophiles.
      - Generate reaction SMILES for every combination between a promising dipole and any synthetic dipolarophile; generate the QM descriptor input for each generated reaction.
      - Use the trained model to predict activation and reaction energies for each reaction SMILES generated in the previous step.
      - Select promising synthetic reactions.
      - Sample a subset of the selected promising synthetic reactions, add the competing native reactions involving the dipoles present in this subset and compute the corresponding reaction profiles.
      - Add the newly computed reaction profiles to the dataset and start the next iteration.
6. Once the dataset is sufficiently enriched with promising bio-orthogonal click reactions, train the model one last time on the final version. Then run through the steps of the active learning loop with relaxed selection criteria and generate final estimates for the bio-orthogonal click potential of the synthetic reactions.

Below, the various auxiliary scripts, directories and repositories developed to (partially) automate this workflow are discussed.

## Search space definition & dataset generation

A separate repository was set up for the definition of the search space and dataset generation. It can be accessed [here](https://github.com/coleygroup/dipolar_cycloaddition_dataset).

The scripts in this repository were also used to validate selected reactions in the active learning loop. 

## QM descriptor generation

A separate repository was set up for the high-throughput computation of QM descriptors. It can be accessed [here](https://github.com/coleygroup/dipolar_cycloaddition_dataset).

## Multitask QM-augmented GNN model

A separate repository was set up for the multitask (QM-augmented) GNN model. It can be accessed [here](https://github.com/tstuyver/multitask_QM_GNN/tree/normal_scaling_ensemble).

## Baseline ML models

All the files related to the baseline models are included in the `baseline_models` directory. The `main.py` script, which runs each of the baseline models sequentially, can be executed as follows:
```
python main.py [--csv-file <path to the file containing the rxn-smiles>] [--input-file <path to the .pkl input file>] [--split-dir <path to the folder containing the requested splits for the cross validation>] [--n-fold <the number of folds to use during cross validation'>]
```
A version of the descriptor input file is included in the `data_files` directory: `input_alt_models.pkl`. To generate an input file from scratch, the `get_descs.py` script can be employed:
```
python get_descs.py --csv-file <path to the file containing the rxn-smiles and targets> --atom-descs-file <path to the .pkl-file containing atom descriptors> --reaction-descs-file <path to the .pkl-file containing reaction descriptors>
```
where the `.csv` file, as well as the atom- and reaction-descriptor files are the outputs of the `get_descriptors.py` script in the [QM_desc_autodE](https://github.com/coleygroup/dipolar_cycloaddition_dataset) repository (vide supra).

## Reaction SMILES generation of the search space

All the files related to reaction SMILES generation can be found in the `rxn_smiles_gen` directory. (explain how to generate both bio and synthetic files etc.)

##  Screening procedure

For each of the screening steps, a script can be found in the `screening_files` directory. Additionally, the `get_pred.py` script, which combines the outputted predictions of the (QM augmented) GNN model with the reaction SMILES from the generated reaction SMILES `.csv` file to yield input files for the screening scripts, can also be found in this repository. This script can be executed as follows:
```
python get_pred.py --iteration <iteration of the active learning loop> --predictions-file <path to the (unprocessed) predictions file> --rxn-smiles-file <path to .csv file containing the reaction SMILES>
```
`extract_promising_dipoles.py` facilitates the selection of promising dipoles, i.e., dipoles which are not too reactive with native dipolarophiles:
```
python extract_promising_dipoles.py --predictions-file <input .csv file containing predicted reaction and activation energies> [--threshold-lower <threshold to decide whether a dipole is too reactive with biofragments>]
```
The retained dipoles -- together with some additional files containing summarizing statistics -- are stored in a newly generated folder, `bio_filter`, as a `.csv` file: `dipoles_above_treshold.csv`. This file can copied into the reaction SMILES generation folder (vide supra) to generate the (pre-filtered) synthetic search space of the active learning loop.

`extract_promising_reactions.py` facilitates the selection of promising synthetic reactions, i.e., selective reactions that are fast under physiological conditions and are more or less irreversible:
```
python extract_promising_reactions.py --predictions-file <input .csv file containing predicted reaction and activation energies> [--threshold-dipolarophiles <threshold for the mean activation energy of a dipolarophile to gauge intrinsic reactivity>] [--threshold-reverse-barrier <threshold for the reverse barrier (to ensure irreversibility)>] [--max-g-act <maximal G_act for a dipole-dipolarophile combination to be considered suitable>]
```
Finally, there are scripts for sampling of the promising reactions for validation. `sample_promising_reactions1.py` was used during the first iteration to select up to 5 promising synthetic reactions involving 1 out of 20 sampled dipoles. The selected synthetic reactions were subsequently complemented with the corresponding biologically inspired reactions involving the same dipoles. This script can be executed as follows:
```
python sample_promising_reactions1.py --promising-reactions-file <.csv file containing predicted reaction and activation energies for the promising reactions> [--number_validation_dipoles <the number of validation dipoles to sample>]
```
`sample_promising_reactions2.py` was used during the second iteration to select all the promising dipoles which weren't considered as part of the first iteration, and for each of these dipoles, all the reactions involving non-cyclooctyne-based dipolarophiles were retained (since cyclooctynes were severly overrepresented in the first iteration). For the dipoles for which less than 5 reactions could be selected in this manner, reactions involving cyclooctyne as the dipolarophile were sampled until 5 reactions were reached. This script can be executed as follows:
```
python sample_promising_reactions2.py --promising-reactions-file <.csv file containing predicted reaction and activation energies for the promising reactions>
``` 


## References

If (parts of) this workflow are used as part of a publication please cite the associated paper:

xxx

Additionally, please cite the paper in which the dataset generation procedure was presented:

xxx

Furthermore, since the workflow makes heavy use of autodE, please also cite the paper in which this code was originally presented:
```
@article{autodE,
  doi = {10.1002/anie.202011941},
  url = {https://doi.org/10.1002/anie.202011941},
  year = {2021},
  publisher = {Wiley},
  volume = {60},
  number = {8},
  pages = {4266--4274},
  author = {Tom A. Young and Joseph J. Silcock and Alistair J. Sterling and Fernanda Duarte},
  title = {{autodE}: Automated Calculation of Reaction Energy Profiles -- Application to Organic and Organometallic Reactions},
  journal = {Angewandte Chemie International Edition}
}
```