## General descriptions
This github repository accompanies manuscript "Reinforcement learning prioritizes general applicability in reaction optimization"

[dataset-analysis](./dataset-analysis) contains analysis functions for all datasets 

[deebo](./deebo) contains all algorithm, 

## Dataset availability
All datasets that were used in this study can be found in a separate [reaction dataset repo](https://github.com/doyle-lab-ucla/ochem-data/tree/main/deebo) by our group.

## Package requirements
Our software is written with minimal dependencies in mind. Only the essential packages are required. 
Here is a list of all the packages that need to be installed to run everything.

- python (3.9.16)
- rdkit (2022.9.3)
- pandas (1.5.1)
- numpy (1.23.4)
- scikit-learn (1.1.3)
- scipy (1.9.3)
- pyyaml (6.0)
- matplotlib (3.7.1)
- tqdm (4.64.1)
- gif (22.11.0)

Version numbers are listed just for reference. 
Installing the exact same version is probably not necessary, 
except for things like `matplotlib` that has changed quite a lot from version to version.

Some of the packages are non-essential, for example, `gif` is only needed if you want to make gifs in `chem_analyze.py`;
if you don't need progress bar, you don't need `tqdm` either. Simply create a conda environment, and all of these packages
can be installed via pip or conda.
