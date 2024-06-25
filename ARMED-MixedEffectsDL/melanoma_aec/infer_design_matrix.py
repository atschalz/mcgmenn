'''
Infer the cluster membership design matrix for the data from unseen batches.
Train a classifier to predict the design matrix from the latent representations
produced by the random effects (domain enhancing) autoencoder. Then, predict on
the latent representations of the unseen batch data.

'''

import numpy as np
import pandas as pd
from armed.misc import expand_data_path, expand_results_path

from armed.tfutils import set_gpu
from armed.models.autoencoder_classifier import ImageClassifier
import tensorflow as tf
import tensorflow.keras.layers as tkl

def load_metadata(path):
    path = expand_data_path(path)
    return pd.read_csv(path, index_col=0)

def load_latents(path):
    path = expand_results_path(path)
    return pd.read_pickle(path)

# Load data
dfMetadataTrain = load_metadata('melanoma/allpdx_selecteddates/data_train.csv')
dfMetadataVal = load_metadata('melanoma/allpdx_selecteddates/data_val.csv')
dfMetadataTest = load_metadata('melanoma/allpdx_selecteddates/data_test.csv')
dfMetadataUnseen = load_metadata('melanoma/allpdx_selecteddates/data_unseen.csv')

strUnseenDataPath = expand_data_path('melanoma/allpdx_selecteddates/data_unseen.npz')
dictUnseen = dict(np.load(strUnseenDataPath))

# Save new dataset with inferred Z here
strUnseenDataOutPath = expand_data_path('melanoma/allpdx_selecteddates/data_unseen_inferred_z.npz')

# Mapping of cluster design matrix columns to dates
arrClasses = dfMetadataTrain['date'].unique()
dictClassToInt = {k: v for v, k in enumerate(arrClasses)}

strTrainDataPath = expand_data_path('melanoma/allpdx_selecteddates/data_train.npz')
strValDataPath = expand_data_path('melanoma/allpdx_selecteddates/data_val.npz')
strTestDataPath = expand_data_path('melanoma/allpdx_selecteddates/data_test.npz')

dictTrain = np.load(strTrainDataPath)
dictVal = np.load(strValDataPath)
dictTest = np.load(strTestDataPath)

layer_in = tkl.Input((256, 256, 1))
layer_out = ImageClassifier(n_clusters=arrClasses.shape[0])(layer_in)
model = tf.keras.Model(layer_in, layer_out)
model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.0001), 
              loss='categorical_crossentropy', metrics='accuracy')
lsCallbacks = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, 
                                               restore_best_weights=True)

model.fit(dictTrain['images'], dictTrain['cluster'],
          validation_data=(dictVal['images'], dictVal['cluster']),
          batch_size=32,
          epochs=10,
          callbacks=lsCallbacks)

arrZUnseen = model.predict(dictUnseen['images'])
dictUnseen['cluster'] = arrZUnseen
np.savez(strUnseenDataOutPath, **dictUnseen)