# Reinforcement learning prioritizes general applicability in reaction optimization
This work has been published in [Nature](https://www.nature.com/articles/s41586-024-07021-y)
[Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.8181283) for this codebase at the time of publication\
[Preprint](https://chemrxiv.org/engage/chemrxiv/article-details/64c7e2e1658ec5f7e5808425) on Chemrxiv


## General descriptions
[dataset-analysis](./dataset-analysis) contains analysis functions for all datasets 

[deebo](./deebo) contains all implemented algorithms, optimization model and testing methods used in the study. 

# Demo Jupyter notebook
- [demo](./deebo/demo.ipynb): general notebook for example usages. 
- [manuscript figures](./deebo/manuscript%20figures.ipynb): necessary functions to reproduce all figures in the manuscript.
- [phenol alkylation](./deebo/demo%20phenol%20akylation.ipynb): the full workflow of the phenol alkylation test reaction. 

## Reaction dataset availability
All datasets that were used in this study can be found in [datasets](./datasets).\
These files are also hosted in a [reaction dataset repo](https://github.com/doyle-lab-ucla/ochem-data/tree/main/deebo) by our group, so they can be streamed and used anywhere with a URL (raw.githubusercontent.com/OWNER/REPO/main/FILEPATH)

## Testing data logs availability
All testing log files can be found in this Zenodo repository [DOI: 10.5281/zenodo.8170874](https://zenodo.org/doi/10.5281/zenodo.8170874)

## Installation requirements
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

### For a step-by-step instruction
1. Download a package management system, such as [conda](https://docs.conda.io/en/latest/).
2. In terminal (or other command line applications), create a conda environment named "bandit" for using this software, specify Python version 3.9 (the version we used during development, probably not necessary):

`conda create --name bandit python=3.9`

3. Activate the conda environment:

`conda activate bandit`

4. Install all external packages deebo requires (`Gif` and `rdkit` are only available from pypi):

*Like discussed above, some of the packages are also not essential. For essential packages required, check the import statements for the scripts containing desired functions or classes that will be used.*

`conda install pandas numpy scikit-learn scipy pyyaml matplotlib tqdm`

`pip install rdkit gif`

5. Download the source code folder from GitHub (by clicking "Download"), or from Zenodo repository 
[DOI: 10.5281/zenodo.8181283](https://zenodo.org/doi/10.5281/zenodo.8181283), 
or with git clone by running:

`git clone https://github.com/doyle-lab-ucla/bandit-optimization.git`

6. Navigate into the source code folder.

for example, if git clone'ed into current directory in the last step, run: 

`cd bandit-optimization/deebo`

7. All functions and classes can be called, for example, via a Jupyter notebook.
Example usage are detailed in [demo.ipynb](./deebo/demo.ipynb).


## Authors
- Jason Y. Wang
- Jason M. Stevens
- Stavros K. Kariofillis
- Mai-Jan Tom
- Dung L. Golden
- Jun Li
- Jose E. Tabora
- Marvin Parasram
- Benjamin J. Shields
- David N. Primer
- Bo Hao
- David Del Valle
- Stacey DiSomma
- Ariel Furman
- G. Greg Zipp
- Sergey Melnikov
- James Paulson
- Abigail G. Doyle*

## What is "deebo"?
Somewhat of a failed acronym attempt for "**d**esign **e**fficient **e**xperiments via **b**andit **o**ptimization"

Originally derived from the name of another optimization model from the Doyle group: [EDBO](https://github.com/b-shields/edbo).

