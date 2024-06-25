# Synthetic datasets

## Setup 
Ensure that the root of this repository is on your `PYTHONPATH` environmental variable. For the digit classifciation example, we also need [Morpho-MNIST](https://github.com/dccastro/Morpho-MNIST) downloaded and on the `PYTHONPATH`. 

See `.env` for an example `PYTHONPATH`. 

## Spiral classification
Classic two-dimensional nonlinear classification benchmark. Data points are sampled along two (or more) spiral functions and the neural network must learn to classify points by their spiral. Random effects are simulated by dividing the data into clusters and random varying the spiral radius in each cluster. 

See `spiral_classification.ipynb` for a comparison of conventional and mixed effects models. Spiral generation parameters can be varied to control the degree of random effects and, optionally, confounding effects.