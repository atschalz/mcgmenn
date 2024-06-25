'''
Main script for training and evaluating classifiers. Performs cross-validation 
using pregenerated data splits and prints the mean and 95% CI of model performance.

Example:
python main.py --model_type mixedeffects --data_dir /path/to/data/splits

See python main.py --help for arguments. 
'''
import os
import glob
import argparse
import numpy as np
import pandas as pd

import tensorflow as tf
from armed.models.cnn_classifier import ImageClassifier, ClusterInputImageClassifier, \
    DomainAdversarialImageClassifier, RandomEffectsClassifier, MixedEffectsClassifier
from armed.misc import expand_data_path, expand_results_path, make_random_onehot
from armed.metrics import classification_metrics

def get_model(model_type: str, n_clusters: int=None):
    tf.random.set_seed(2343)
    if model_type == 'conventional':    
        model = ImageClassifier()
        model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.0001),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=[tf.keras.metrics.AUC(name='auroc')])
         
    elif model_type == 'clusterinput':
        model = ClusterInputImageClassifier()
        model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.0001),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=[tf.keras.metrics.AUC(name='auroc')])
    
    elif model_type == 'adversarial':
        model = DomainAdversarialImageClassifier(n_clusters=n_clusters)
        model.compile(opt_adversary=tf.keras.optimizers.Nadam(lr=0.0001),
                    opt_classifier=tf.keras.optimizers.Nadam(lr=0.0001),
                    metric_classifier=tf.keras.metrics.AUC(name='auroc'))
    
    elif model_type == 'randomeffects':
        model = RandomEffectsClassifier(intercept_post_init_scale=0.1,
                                        intercept_prior_scale=0.25)
        model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.0001),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=[tf.keras.metrics.AUC(name='auroc')])
        
    elif model_type == 'mixedeffects':
        model = MixedEffectsClassifier(n_clusters=n_clusters,
                                    intercept_post_init_scale=0.1,
                                    intercept_prior_scale=0.25)
        model.compile(opt_adversary=tf.keras.optimizers.Nadam(lr=0.0001),
                    opt_classifier=tf.keras.optimizers.Nadam(lr=0.0001),
                    metric_classifier=tf.keras.metrics.AUC(name='auroc'))
    else:
         raise ValueError(model_type, 'not recognized')           
    return model

def train_evaluate(split_dir: str, model_type: str, epochs: int=20, weights_path:str = None,
                   verbose: int=0, randomize_z=False):

    dictDataTrain = np.load(os.path.join(split_dir, 'data_train.npz'))
    dictDataVal = np.load(os.path.join(split_dir, 'data_val.npz'))
    dictDataTest = np.load(os.path.join(split_dir, 'data_test.npz'))
    dictDataUnseen = np.load(os.path.join(split_dir, 'data_unseen.npz'))

    # Weight each class by 1 - class frequency
    dictClassWeights = {0.: dictDataTrain['label'].mean(),
                        1.: 1 - dictDataTrain['label'].mean()}

    if model_type == 'conventional':    
        train_in = dictDataTrain['images']
        val_in = dictDataVal['images']
        test_in = dictDataTest['images']
        unseen_in = dictDataUnseen['images']
    else:
        train_in = (dictDataTrain['images'], dictDataTrain['cluster'])
        val_in = (dictDataVal['images'], dictDataVal['cluster'])
        test_in = (dictDataTest['images'], dictDataTest['cluster'])
        unseen_in = (dictDataUnseen['images'], dictDataUnseen['cluster'])
    
    model = get_model(model_type, n_clusters=dictDataTrain['siteorder'].shape[0])
        
    lsCallbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_auroc', mode='max',
                                                    patience=5, 
                                                    restore_best_weights=True)]
        
    model.fit(x=train_in,
            y=dictDataTrain['label'],
            batch_size=32,
            epochs=epochs,
            verbose=verbose,
            class_weight=dictClassWeights,
            callbacks=lsCallbacks,
            validation_data=(val_in, dictDataVal['label']))

    arrPredTrain = model.predict(train_in, verbose=0)
    arrPredVal = model.predict(val_in, verbose=0)
    
    # Make random Z inputs for test and unseen site data
    if randomize_z:
        nClusters = len(dictDataTrain['siteorder'])
        arrImagesTest = test_in[0]
        nTest = arrImagesTest.shape[0]
        arrZTest = make_random_onehot(nTest, nClusters)
        test_in = (arrImagesTest, arrZTest)
    
        arrImagesUnseen = unseen_in[0]
        nUnseen = arrImagesUnseen.shape[0]
        arrZUnseen = make_random_onehot(nUnseen, nClusters)    
        unseen_in = (arrImagesUnseen, arrZUnseen)
    
    arrPredTest = model.predict(test_in, verbose=0)
    arrPredUnseen = model.predict(unseen_in, verbose=0)
    dictMetricsTrain, youden = classification_metrics(dictDataTrain['label'], arrPredTrain, 
                                                      fixed_sens=0.7, fixed_spec=0.7)
    dictMetricsVal, _ = classification_metrics(dictDataVal['label'], arrPredVal, 
                                               fixed_sens=0.7, fixed_spec=0.7)
    dictMetricsTest, _ = classification_metrics(dictDataTest['label'], arrPredTest, 
                                                fixed_sens=0.7, fixed_spec=0.7)
    dictMetricsUnseen, _ = classification_metrics(dictDataUnseen['label'], arrPredUnseen, 
                                                  fixed_sens=0.7, fixed_spec=0.7)
        
    lsMetrics = [dictMetricsTrain, dictMetricsVal, dictMetricsTest, dictMetricsUnseen]
    
    dfMetrics = pd.DataFrame(lsMetrics)
    dfMetrics['partition'] = ['Train', 'Val', 'Test', 'Unseen']
    
    if weights_path:
        model.save_weights(weights_path)
    
    return dfMetrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='ADNI23_sMRI/right_hippocampus_slices_2pctnorm/coronal_MNI-6_numpy/12sites',
                        help='Path to directory containing data splits.')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model_type', type=str, choices=['conventional', 'clusterinput', 'adversarial', 
                                                           'mixedeffects', 'randomeffects'],
                        required=True, help='Model type.')
    parser.add_argument('--epochs', type=int, default=20, help='Training duration. Defaults to 20')
    parser.add_argument('--randomize_sites', action='store_true', help='Use a randomized site membership'
                        ' input on test and unseen site data (RE ablation test).')
    parser.add_argument('--gpu', type=int, help='GPU to use. Defaults to all.')
    parser.add_argument('--verbose', type=int, default=0, help='Show training progress.')

    args = parser.parse_args()
    
    if args.gpu:
        from armed.tfutils import set_gpu
        set_gpu(args.gpu)
    
    strDataDir = expand_data_path(args.data_dir)
    lsSplitDirs = glob.glob(os.path.join(strDataDir, 'split*'))
    lsSplitDirs.sort()

    strOutDir = expand_results_path(args.out_dir, make=True)

    lsAllMetrics = []
    for strSplitDir in lsSplitDirs:
        strSplitName = os.path.basename(strSplitDir)
        print(strSplitName)
        strOutPath = os.path.join(strOutDir, strSplitName + '_weights.h5')
        df = train_evaluate(strSplitDir, args.model_type, weights_path=strOutPath, epochs=args.epochs, 
                            verbose=args.verbose, randomize_z=args.randomize_sites)
        df['split'] = os.path.basename(strSplitDir)
        lsAllMetrics += [df]
        
    dfAllMetrics = pd.concat(lsAllMetrics)
    dfAllMetrics.to_csv(os.path.join(strOutDir, 'metrics.csv'))
    
    dfMean = dfAllMetrics.groupby('partition').mean()
    dfSE = dfAllMetrics.groupby('partition').std() / (len(lsSplitDirs) ** 0.5)
    df95CILow = dfMean - dfSE * 1.96
    df95CIHi = dfMean + dfSE * 1.96
    dfMeanCI = pd.concat({'Mean': dfMean, '95CI Low': df95CILow, '95CI Hi': df95CIHi}, axis=1)
    print(dfMeanCI.loc[['Train', 'Val', 'Test', 'Unseen']].to_string())