'''
Call a trained model to produce latent representations, reconstructions, or classifications.
'''
import os
import argparse
import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf

from armed.tfutils import set_gpu
from armed.misc import expand_data_path, expand_results_path, make_random_onehot
from armed.models.autoencoder_classifier import load_weights_base_aec

from main import _get_model

parser = argparse.ArgumentParser()
parser.add_argument('--image_data', type=str, required=True, 
                    help='Path to .npz file containing images.')
parser.add_argument('--image_list', type=str, required=True, 
                    help='Path to CSV table containing image paths and metadata.')
parser.add_argument('--weights', type=str, default=None, required=True,help='Saved model weights')
parser.add_argument('--randomize_batch', action='store_true', help='Use a randomized batch membership'
                    ' input when testing (as an ablation test).')
parser.add_argument('--model_type', type=str, 
                    choices=['conventional', 'adversarial', 'mixedeffects', 'randomeffects'], 
                    default='conventional', help='Model type')
parser.add_argument('--gpu', type=int, help='GPU to use. Defaults to all.')

parser.add_argument('--latents', type=str, help='Compute latents and save to this .pkl file')
parser.add_argument('--reconstructions', type=str, help='Compute reconstructions and save to this .npy file')
parser.add_argument('--classifications', type=str, help='Compute classification predictions and save to this .csv file')


args = parser.parse_args()

if args.gpu:
    set_gpu(args.gpu)

strWeightsPath = expand_results_path(args.weights)
print('Loading model weights from', strWeightsPath, flush=True)

dictData = np.load(expand_data_path(args.image_data))
dfImages = pd.read_csv(expand_data_path(args.image_list), index_col=0)
nClusters = dictData['cluster'].shape[1]

model = _get_model(args.model_type, n_clusters=nClusters)

if args.model_type == 'conventional':        
    data_in = dictData['images']
else:
    z = dictData['cluster']
    if args.randomize_batch:
        z = make_random_onehot(z.shape[0], z.shape[1])
    data_in = (dictData['images'], z)

# Model must be compiled and called once before loading weights
model.compile()
_ = model.predict(data_in, steps=1, batch_size=32)    

if args.model_type == 'conventional':
    load_weights_base_aec(model, strWeightsPath)
else:
    model.load_weights(strWeightsPath, by_name=True, skip_mismatch=False)

if (args.latents is not None) | (args.classifications is not None):
    # Isolate the encoder 
    if args.model_type in ['mixedeffects', 'randomeffects']:
        # encoder takes X and Z inputs
        encoder_in = (tf.keras.layers.Input(dictData['images'].shape[1:]),
                    tf.keras.layers.Input(nClusters))
    else:
        # encoder takes X input
        encoder_in = tf.keras.layers.Input(dictData['images'].shape[1:])

    encoder_out = model.encoder(encoder_in)
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[-1]
        
    encoder = tf.keras.Model(encoder_in, encoder_out)
    
    # Compute classifications
    if args.classifications:
        strClassificationOutPath = expand_results_path(args.classifications)
        
        # Connect the classifier to the encoder
        if args.model_type == 'mixedeffects':
            classifier_out = model.classifier((encoder_out, encoder_in[1]))
        elif args.model_type == 'randomeffects':
            classifier_out = model.latent_classifier(encoder_out)[0]
        else:
            classifier_out = model.classifier(encoder_out)
            
        classifier = tf.keras.Model(encoder_in, classifier_out)
        pred = classifier.predict(data_in, verbose=1)       

        dfPred = pd.DataFrame({'true': dfImages['met-eff'].values == 'low', 
                            'pred': pred.squeeze()}, 
                            index=dfImages['image'].values)
        dfPred.to_csv(strClassificationOutPath)
    
    # Compute latent representations
    if args.latents:
        strLatentsOutPath = expand_results_path(args.latents)
            
        encoder_output = encoder.predict(data_in, verbose=1)
        if isinstance(encoder_output, tuple):
            arrLatents = encoder_output[-1]
        else:
            arrLatents = encoder_output

        dfLatents = pd.DataFrame(arrLatents, index=dfImages['image'].values)
        dfLatents.to_pickle(strLatentsOutPath)
    
# Compute reconstructions
if args.reconstructions:
    strReconsOutPath = expand_results_path(args.reconstructions)
    
    # Break data into batches to avoid memory overflow
    lsRecons = []
    if isinstance(data_in, tuple):
        nImages = data_in[0].shape[0]
    else:
        nImages = data_in.shape[0]
    nBatches = int(np.ceil(nImages / 1000))

    for iBatch in tqdm.tqdm(range(nBatches)):
        iStart = 1000 * iBatch
        iEnd = np.min([1000 * (iBatch + 1), nImages])
        
        if isinstance(data_in, tuple):
            batch_in = (data_in[0][iStart:iEnd,], data_in[1][iStart:iEnd,])
        else:
            batch_in = data_in[iStart:iEnd,]
            
        arrRecons = model.predict(batch_in, batch_size=32)[0]        
        lsRecons += [arrRecons]
        
    arrRecons = np.concatenate(lsRecons, axis=0)
    np.save(strReconsOutPath, arrRecons)

