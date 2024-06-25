# MC-GMENN: Enabling Mixed Effects Neural Networks for Diverse, Clustered Data Using Monte Carlo Methods

This repository contains the code to reproduce the experiments in our IJCAI 2024 paper. In the future, the MC-GMENN model will also be made available as a pip package.

To replicate the experiments reported in the paper and in the supplementary material, first Run "data/download_pargent2022_datasets.py" to download the datasets used in the high-cardinality categorical features benchmark study. 
Afterwards, the notebooks can be executed to reproduce the results. Note that the GPU configurations in the first cell of each notebook might need to be adjusted for the user-specific setup.

The notebooks folder contains examples of how to use MC-GMENN on synthetic as well as real data along with benchmark models to evaluate the performance of MC-MENN.

The repository also contains a copy of LMMNN (https://github.com/gsimchoni/lmmnn) with the following changes made:
- function reg_nn_lmmnn in nn.py changed to return b_hat and the model itself
- adapt binary data generation function to return b_hat
- function reg_nn_lmmnn in nn.py changed to include option to pass a base model, optimizer and validation data as a parameter to allow easier comparison to other methods 

Furthermore, a copy of ARMED (https://gitfront.io/r/DeepLearningForPrecisionHealthLab/54f18307815dfb2148fbc2d14368c1268b63825e/ARMED-MixedEffectsDL/) is included, where we made additions be able to run random intercept models with the base neural networks described in our paper.

Requirements:
- numpy
- pandas
- keras
- tensorflow
- tensorflow_addons
- tensorflow_probability
- keras-vis
- torch
- category_encoders
- matplotlib
- scikit-learn
- scipy
- vis
- pickle
- pyyaml
- gc