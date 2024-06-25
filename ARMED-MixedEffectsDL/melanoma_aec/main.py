'''
Main script for training and evaluating an autoencoder-classifier (AEC). 

Required arguments include the model type and output directory. 

To train a model:
python main.py --model_type conventional --output_dir /path/to/output/location

To evaluate the model after training:
python main.py --model_type conventional --output_dir /path/to/output/location --do_test 
    --load_weights_epoch <epoch index to load weights from>

See python main.py --help for all arguments. 
'''

import os
import argparse
import json
import glob
import numpy as np
import pandas as pd

from armed.misc import expand_data_path, expand_results_path, make_random_onehot

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

def _shuffle_data(data_dict):
    # shuffle samples
    arrIdx = np.arange(data_dict['images'].shape[0])
    np.random.seed(64)
    np.random.shuffle(arrIdx)
    return {k: v[arrIdx,] for k, v in data_dict.items()}

def _random_samples(data_dict, metadata=None, n=100):
    # select n random samples
    arrIdx = np.arange(data_dict['images'].shape[0])
    np.random.seed(64)
    arrSampleIdx = np.random.choice(arrIdx, size=n)
    dictNew = {k: v[arrSampleIdx,] for k, v in data_dict.items()}
    if metadata is not None:
        return dictNew, metadata.iloc[arrSampleIdx,]
    else:
        return dictNew

def _get_model(model_type, n_clusters=10):
    # Build and compile a model with some preset hyperparameters
    import tensorflow as tf
    from armed.models import autoencoder_classifier
    if model_type == 'conventional':        
        model = autoencoder_classifier.BaseAutoencoderClassifier(n_latent_dims=56)
          
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                      loss=[tf.keras.losses.MeanSquaredError(name='mse'),
                            tf.keras.losses.BinaryCrossentropy(name='bce')],
                      loss_weights=[1.0, 0.1],
                      metrics=[[],
                               [tf.keras.metrics.AUC(name='auroc')]])  
        
    elif model_type == 'adversarial':
        model = autoencoder_classifier.DomainAdversarialAEC(n_clusters=n_clusters, n_latent_dims=56)
        
        model.compile(loss_recon=tf.keras.losses.MeanSquaredError(),
                      loss_class=tf.keras.losses.BinaryCrossentropy(),
                      loss_adv=tf.keras.losses.BinaryCrossentropy(),
                      metric_class=tf.keras.metrics.AUC(name='auroc'),
                      metric_adv=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                      opt_autoencoder=tf.keras.optimizers.Adam(lr=0.0001),
                      opt_adversary=tf.keras.optimizers.Adam(lr=0.0001),
                      loss_recon_weight=1.0,
                      loss_class_weight=0.01,
                      loss_gen_weight=0.1)
    
    elif model_type == 'mixedeffects':
        model = autoencoder_classifier.MixedEffectsAEC(n_clusters=n_clusters, n_latent_dims=56)
        model.compile()
        
    elif model_type == 'randomeffects':
        model = autoencoder_classifier.DomainEnhancingAutoencoderClassifier(n_clusters=n_clusters, n_latent_dims=56, kl_weight=1e-7)
        model.compile()
        
    return model

def train_model(model_type: str, 
                data_train: dict, 
                data_val: dict, 
                train_metadata: pd.DataFrame,
                val_metadata: pd.DataFrame,
                output_dir: str,
                epochs: int=10,
                verbose: bool=True,
                ):
    """Train a model.

    Args:
        model_type (str): name of model type
        data_train (dict): training data, should have keys 'images', 'label',
            and 'cluster'
        data_val (dict): validation data, should have keys 'images', 'label',
            and 'cluster'
        train_metadata (pd.DataFrame): training metadata
        val_metadata (pd.DataFrame): validation metadata
        output_dir (str): path to output location
        epochs (int, optional): epochs to train. Defaults to 10.
        verbose (bool, optional): training verbosity. Defaults to True.

    Returns:
        dict: final model metrics
    """    

    # Imports done inside function so that memory is allocated properly when
    # used with Ray Tune    
    import tensorflow as tf
    import tensorflow.keras.layers as tkl
    from armed.callbacks import aec_callbacks
        
    strOutputDir = expand_results_path(output_dir)

    model = _get_model(model_type, n_clusters=data_train['cluster'].shape[1])
    if model_type == 'conventional':        
        train_in = data_train['images']
        train_out = (data_train['images'], data_train['label'])
        val_in = data_val['images']
        val_out = (data_val['images'], data_val['label'])
          
    else:
        train_in = (data_train['images'], data_train['cluster'])
        train_out = (data_train['images'], data_train['label'])
        val_in = (data_val['images'], data_val['cluster'])
        val_out = (data_val['images'], data_val['label'])

    # Get a few samples to generate example reconstructions every epoch
    data_sample = _random_samples(data_val, n=8)
    arrBatchX = data_sample['images']
    arrBatchZ = data_sample['cluster']
    
    # Callbacks:
    # Create figure with example reconstructions
    recon_images = aec_callbacks.make_recon_figure_callback(arrBatchX, model, output_dir,
                                            clusters=None if model_type == 'conventional' else arrBatchZ,
                                            mixedeffects=model_type == 'mixedeffects')
    # Compute image metrics 
    compute_image_metrics = aec_callbacks.make_image_metrics_callback(model, val_in, val_metadata, output_dir,
                                            output_idx=1 if model_type == 'mixedeffects' else 0)
    
    lsCallbacks = [tf.keras.callbacks.CSVLogger(os.path.join(strOutputDir, 'training_log.csv')),
                   tf.keras.callbacks.LambdaCallback(on_epoch_end=recon_images),
                   tf.keras.callbacks.LambdaCallback(on_epoch_end=compute_image_metrics),
                   tf.keras.callbacks.ModelCheckpoint(os.path.join(strOutputDir, 'epoch{epoch:03d}_weights.h5'),
                                                      save_weights_only=True)]
    
    # Isolate the encoder
    if model_type == 'randomeffects':
        # RE model takes both image and cluster as inputs
        encoder_in = (tkl.Input((256, 256, 1), name='encoder_in_x'),
                      tkl.Input((data_train['cluster'].shape[1],), name='encoder_in_z'))
        encoder_out = model.encoder(encoder_in)
        encoder_data = train_in
    else:
        encoder_in = tkl.Input((256, 256, 1), name='encoder_in')
        encoder_out = model.encoder(encoder_in)
        if isinstance(encoder_out, tuple):
            # If the encoder outputs all layer activations, keep only the latent rep output
            encoder_out = encoder_out[-1]
        encoder_data = data_train['images']
        
    encoder = tf.keras.models.Model(encoder_in, encoder_out, name='standalone_encoder')
    
    # Create callback to save latent representations for training data every epoch
    compute_latents = aec_callbacks.make_compute_latents_callback(encoder, encoder_data,
                                                                  train_metadata, output_dir)
    lsCallbacks += [tf.keras.callbacks.LambdaCallback(on_epoch_end=compute_latents)]
    
    # Train
    history = model.fit(train_in, train_out,
                        epochs=epochs,
                        verbose=verbose,
                        batch_size=16, # changed from 32 to fit on P40
                        validation_data=(val_in, val_out),
                        shuffle=True,
                        callbacks=lsCallbacks)
    # Get final metrics
    dfHistory = pd.DataFrame(history.history)
    dictResults = dfHistory.iloc[-1].to_dict()
    
    # Compute clustering metrics on latents
    arrLatents = encoder.predict(encoder_data)
    arrLatents -= arrLatents.mean(axis=0)
    arrLatents /= arrLatents.std(axis=0)    
    db = davies_bouldin_score(arrLatents, train_metadata['date'])
    ch = calinski_harabasz_score(arrLatents, train_metadata['date'])
    
    dictResults.update(db=db, ch=ch)
    
    return dictResults


def test_model(model_type: str, 
               saved_weights: str,
               data: dict,
               randomize_z: bool = False):
    """Evaluate trained model.

    Args:
        model_type (str): name of model type
        saved_weights (str): path to saved weights in .h5 file
        data (dict): data for evaluating model, should have keys 'images',
            'label', and 'cluster'
        randomize_z (bool): randomize the cluster membership input as an 
            ablation test. Defaults to False.

    Returns:
        dict: model metrics
    """    
    
    # Imports done inside function so that memory is allocated properly when
    # used with Ray Tune    
    from armed.models.autoencoder_classifier import load_weights_base_aec
        
    data = _shuffle_data(data)

    model = _get_model(model_type, n_clusters=data['cluster'].shape[1])
    if model_type == 'conventional':
        data_in = data['images']
        data_out = (data['images'], data['label'])
        
    else:
        z = data['cluster']
        if randomize_z:
            z = make_random_onehot(z.shape[0], z.shape[1])
        
        data_in = (data['images'], z)
        data_out = (data['images'], data['label'])
    
    # Call model once to instantiate weights
    _ = model.predict(data_in, steps=1, batch_size=32)    
    if model_type == 'conventional':
        # Workaround for weight loading bug
        load_weights_base_aec(model, saved_weights)
    else:
        model.load_weights(saved_weights)
            
    dictMetrics = model.evaluate(data_in, data_out, 
                                 batch_size=32,
                                 return_dict=True)
                
    return dictMetrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_metadata', type=str, 
                        default='melanoma/allpdx_selecteddates/data_train.csv',
                        help='Path to CSV table containing training image metadata.')
    parser.add_argument('--training_data', type=str, 
                        default='melanoma/allpdx_selecteddates/data_train.npz',
                        help='Path to .npz file containing training images.')
    parser.add_argument('--val_metadata', type=str, 
                        default='melanoma/allpdx_selecteddates/data_val.csv',
                        help='Path to CSV table containing validation image metadata.')
    parser.add_argument('--val_data', type=str, 
                        default='melanoma/allpdx_selecteddates/data_val.npz',
                        help='Path to .npz file containing validation images.')
    parser.add_argument('--test_metadata', type=str, 
                        default='melanoma/allpdx_selecteddates/data_test.csv',
                        help='Path to CSV table containing test image metadata.')
    parser.add_argument('--test_data', type=str, 
                        default='melanoma/allpdx_selecteddates/data_test.npz',
                        help='Path to .npz file containing test images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('--model_type', type=str, choices=['conventional', 'adversarial', 'mixedeffects', 'randomeffects'],
                        required=True, help='Model type.')
    parser.add_argument('--epochs', type=int, default=10, help='Training duration. Defaults to 10')
    parser.add_argument('--do_test', action='store_true', help='Evaluate on test')
    parser.add_argument('--randomize_batch', action='store_true', help='Use a randomized batch membership'
                        ' input when testing (as an ablation test).')
    parser.add_argument('--load_weights_epoch', type=int, default=None, help='If evaluating on test, load weights'
                        ' from this epoch and skip training.')
    parser.add_argument('--verbose', type=int, default=1, help='Show training progress.')
    parser.add_argument('--gpu', type=int, help='GPU to use. Defaults to all.')
    parser.add_argument('--smoketest', action='store_true', 
                        help='For quick testing purposes, use a dataset of 100 samples')

    args = parser.parse_args()

    if args.gpu:
        # Select GPU to use
        from armed.tfutils import set_gpu
        set_gpu(args.gpu)
        
    strOutputDir = expand_results_path(args.output_dir, make=True)
    
    if args.load_weights_epoch is None:
        # If no weights were selected to load, train model 
        strTrainDataPath = expand_data_path(args.training_data)
        strTrainMetaDataPath = expand_data_path(args.training_metadata)
        strValDataPath = expand_data_path(args.val_data)
        strValMetaDataPath = expand_data_path(args.val_metadata)
        dfTrainMetadata = pd.read_csv(strTrainMetaDataPath, index_col=0)
        dfValMetadata = pd.read_csv(strValMetaDataPath, index_col=0)

        dictDataTrain = np.load(strTrainDataPath)
        dictDataVal = np.load(strValDataPath)
        
        if args.smoketest:
            dictDataTrain, dfTrainMetadata = _random_samples(dictDataTrain, dfTrainMetadata, n=100)
            dictDataVal, dfValMetadata = _random_samples(dictDataVal, dfValMetadata, n=100)
    
        dictMetrics = train_model(model_type=args.model_type, 
                                data_train=dictDataTrain, 
                                data_val=dictDataVal,
                                train_metadata=dfTrainMetadata, 
                                val_metadata=dfValMetadata,
                                output_dir=strOutputDir,
                                epochs=args.epochs,
                                verbose=args.verbose == 1)
        
        print(dictMetrics)
        
    if args.do_test:
        strTestDataPath = expand_data_path(args.test_data)
        strTestMetaDataPath = expand_data_path(args.test_metadata)
        dictDataTest = np.load(strTestDataPath)
        dfTestMetadata = pd.read_csv(strTestMetaDataPath, index_col=0)
        
        if args.smoketest:
            dictDataTest, dfTestMetadata = _random_samples(dictDataTest, dfTestMetadata, n=100)

        if args.load_weights_epoch is not None:
            strSavedWeightsPath = os.path.join(strOutputDir, f'epoch{args.load_weights_epoch:03d}_weights.h5')
            assert os.path.exists(strSavedWeightsPath)
        else:
            # Grab the last epoch weights
            lsWeights = glob.glob(os.path.join(strOutputDir, '*weights.h5'))
            lsWeights.sort()
            strSavedWeightsPath = lsWeights[-1]
            
        print('Loading weights from', strSavedWeightsPath, flush=True)

        dictMetrics = test_model(model_type=args.model_type,
                                 saved_weights=strSavedWeightsPath,
                                 data=dictDataTest,
                                 randomize_z=args.randomize_batch)
        
        with open(os.path.join(strOutputDir, 'test_metrics.json'), 'w') as f:
            json.dump(dictMetrics, f, indent=4)