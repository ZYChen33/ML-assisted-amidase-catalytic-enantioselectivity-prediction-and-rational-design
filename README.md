# Machine learning-assisted amidase-catalytic enantioselectivity prediction and rational design of variants for improving enantioselectivity

This repository contains the code and data used in "Machine learning-assisted amidase-catalytic enantioselectivity prediction and rational design of variants for improving enantioselectivity". It includes tools for generating descriptors, training models, and performing virtual screening.
[![DOI](https://zenodo.org/badge/840572472.svg)](https://zenodo.org/doi/10.5281/zenodo.13759700)

## Related Publication
For more details, please refer to our published paper: [DOI: 10.1038/s41467-024-53048-0](https://doi.org/10.1038/s41467-024-53048-0).

## Requirements

To run the scripts provided in this repository, you'll need the following Python libraries:
- Python  3.8+
- scikit-learn >= 1.0.2
- pandas >= 1.4.3
- numpy >= 1.24.4
- matplotlib >= 3.5.2
- imbalanced_learn >= 0.12.2
- rdkit >= 2023.9.6
- sklearn_relief >= 1.0.0b2
- scipy >= 1.10.1

## Installation Guide

You can install the required Python packages using the following commands: 
pip install <package_name>==<version_number>

  
For example, to install scikit-learn version 1.0.2, you would run:
pip install scikit-learn==1.0.2

Ensure you replace <package_name> with the name of the required library and <version_number> with the specific version you need. It's important to use the correct versions to ensure compatibility with the script.

### Estimated Installation Time:

The installation process should take around 5-10 minutes on a standard desktop computer with a stable internet connection.

## Repository Structure

The repository is divided into two main parts:
- **Descriptors Folder:** This folder contains the code required for generating molecular descriptors and a small sample molecule for testing.
- **Main Code:** This section contains  the primary machine learning and virtual screening code. The `90.py` script uses a criterion of $\Delta \Delta G^{\neq} = 2.40$ kcal/mol as an example. The necessary data files for training this model and virtual screening are also provided. Users can modify the hyperparameters and criterion in the script to tailor the model to their specific dataset.

## Usage

### Downloading and Setting Up

1. **Download the Entire Repository**

   To get started, download the entire repository by:
   - Clicking the `Code` button at the top right of the page, then selecting `Download ZIP`.
   - After downloading, unzip the file to a directory where you want to store the code and data.

2. **Prepare Data and Scripts**

   - After unzipping, you will find all necessary scripts and sample data in the repository. Replace the sample data files with your own datasets, ensuring they match the required format.
   - If you need to change the data file paths, adjust the relative paths in the corresponding Python scripts. By default, the scripts will read data files from the current working directory.

### Generating Descriptors

To generate molecular descriptors:

1. Navigate to the `Descriptors` directory.
2. Follow the methodology described in the associated article to generate descriptors and use the sample molecule files provided in the `Descriptors` folder to create your initial input files.
3. Run the scripts in the correct order: run `1_` scripts before `2_` scripts.
These scripts will process your molecular data and output a set of descriptors suitable for machine learning models.

### Training ML Models and Virtual Screening

The main machine learning code is contained in the `90.py` script, which uses a criterion of $\Delta \Delta G^{\neq} = 2.40$ kcal/mol as an example. This script will train models using the provided data and can be adapted to perform virtual screening on new molecular variants.

### Expected Run Time

Depending on the dataset size, the hyperparameter optimization of a model usually takes from a few hours to a day, and the enantioselectivity prediction toward one substrate usually takes a few seconds on a standard desktop computer.
