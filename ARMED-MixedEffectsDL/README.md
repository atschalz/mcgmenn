# Adversarially-regularized mixed effects deep learning (ARMED) 
A framework for building mixed effects neural networks for clustered and non-iid data, as reported in our paper:

Nguyen KP, Treacher A, Montillo A. "Adversarially-regularized mixed effects deep learning (ARMED) models improve interpretability, performance, and generalization on clustered (non-iid) data." *in press at IEEE Transactions on Pattern Analysis and Machine Intelligence*. 

Our preprint is available at http://arxiv.org/abs/2202.11783v1.

## Setup 
Global directory paths should be editted in `armed.settings`:

1. `RESULTSDIR`: where experimental results will be stored
2. `DATADIR`: where downloaded and simulated datasets are stored

Add the repository root to the `PYTHONPATH`. If using Visual Studio Code, this can be done by modifying the `.env` file, which is read by the Python extension when running any code interactively. 

## Dependencies
See `conda_environment.yml` for Python dependencies. A new environment with these dependencies can be created using
```
conda env create -f conda_environment.yml --prefix </path/to/environment/location>
```

## Table of contents
The main [`armed`](./armed) package contains the general-purpose tools for building ARMED models. The random effects layers can be found in `armed.models.random_effects`. Below are links to specific applications of ARMED models included in the above manuscript. 

* [Synthetic datasets](./synthetic_dataset): dense feedforward neural network applied to simulated spiral classification problems with random effects
* [MCI conversion](./ad_conversion): dense feedforward neural network applied to classification of stable vs. progressive mild cognitive impairment
* [AD diagnosis](./adni_t1w): convolutional neural network applied to classification of Alzheimer's Disease vs. cognitively normal subjects from T1w MRI
* [Melanoma cell image compression and classification](./melanoma_aec): convolutional autoencoder applied to compression of melanoma live cell images with simultaneous phenotype classification