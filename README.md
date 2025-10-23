# Max-Affine-Data
Contains all code and DFT data used to train models and generate plots/predictions for the paper "Interpretable Machine Learning Model Captures Energy Landscape of all Materials with Seven Numbers per Element".

To get training data from the Materials Project, use extract_material_data.py. Subsequently, to initialize features for model training, use initialize_element_features.py. To train the max-affine model and neural network, use train_MA.py and train_NN.py, respectively. The models folder contains the fully trained max-affine model and neural network used to make predcitions in the paper. 

All ab-initio calculation data for defect energy calculations is in the dft folder. The DFT output files contain all species and structural information, as well as energy/density cutoff and pseudopotential information. To obtain interstitial energies, we used the energy from the most stable (lowest energy) of the tetrahedral and octahedral interstitial sites. We also did this for select oxide vacancy energies. 

The plot_defects_HEP.ipynb notebook predicts and plots all defect energies and high entropy phosphide elemental correlations. The periodic_table_facets.ipynb creates the periodic table image in the paper using the max-affine model.
