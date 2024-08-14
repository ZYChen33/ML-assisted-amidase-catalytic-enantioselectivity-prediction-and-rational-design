# ML-assisted amidase-catalytic enantioselectivity prediction and rational design of variants for improving enantioselectivity
This repository contains the primary machine learning (ML) code used in "ML-assisted Amidase-Catalytic Enantioselectivity Prediction and Rational Design of Variants for Improving Enantioselectivity." The code includes key components for model training and virtual screening. 

The 90.py uses a criterion of \(\Delta \Delta G^{\neq}\)= 2.40 kcal/mol as an example. The required CSV files for training and virtual screening are also included in this repository. Users can modify the hyperparameters and criterion in the script to adapt the model to their specific dataset. 

## Requirements
- Python = 3.8
To run the script, you'll need the following Python libraries:
- scikit-learn = 1.0.2
- pandas = 1.4.3
- numpy = 1.24.4
- matplotlib = 3.5.2
- imbalanced_learn = 0.12.2
- rdkit = 2023.9.6
- sklearn_relief = 1.0.0b2
  
You can install them using:
pip install <package_name>==<version_number>
For example, to install scikit-learn version 1.0.2, you would run:
pip install scikit-learn==1.0.2
Ensure you replace <package_name> with the name of the required library and <version_number> with the specific version you need. It's important to use the correct versions to ensure compatibility with the script.

