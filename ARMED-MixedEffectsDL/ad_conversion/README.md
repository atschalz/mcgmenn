# Classification of stable vs. progressive mild cognitive impairment (MCI).
This application compares various models for the classification of stable vs. progressive MCI. Progressive MCI is defined as conversion to dementia within 24 months. Data was sourced from the ADNI dataset, specifically the ADNIMERGE preprocessed and pre-curated table. 

## 1. Partition data into nested k-folds
Run `create_kfolds.py` to divide data into 10x10 nested stratified k-folds. 

## 2. Univariate analyses
`univariate_analyses.ipynb` performs some traditional statistical analyses to measure the influence of site in each feature. 

## 3. Deep learning models
`model_comparison.ipynb` compares a conventional neural network with domain adversarial and mixed effects models. 
`model_comparison_probe_features.ipynb` adds synthetic confounded features to the dataset and compares each model's ability to separate truly informative from known confounded features, in terms of feature importance.