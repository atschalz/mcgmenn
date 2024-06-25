'''
Use a CNN, trained to predict site from an image, to infer the cluster
membership design matrix Z for each image from an unseen site.
'''

import os
import numpy as np
from armed.tfutils import set_gpu

set_gpu(1, 0.5)

import tensorflow as tf
import tensorflow.keras.layers as tkl
from armed.misc import expand_data_path

strDataDir = expand_data_path('ADNI23_sMRI/right_hippocampus_slices_2pctnorm/coronal_MNI-6_numpy/12sites')

dictDataTrain = np.load(os.path.join(strDataDir, 'data_train.npz'))
dictDataVal = np.load(os.path.join(strDataDir, 'data_val.npz'))
# dictDataTest = np.load(os.path.join(strDataDir, 'data_test.npz'))
dictDataUnseen = np.load(os.path.join(strDataDir, 'data_unseen.npz'))

nClusters = dictDataTrain['siteorder'].shape[0]

# Simple CNN classifier
x = tkl.Input(dictDataTrain['images'].shape[1:])
h = tkl.Conv2D(64, 3, padding='same')(x)
h = tkl.BatchNormalization()(h)
h = tkl.PReLU()(h)
h = tkl.MaxPool2D()(h)

h = tkl.Conv2D(64, 3, padding='same')(h)
h = tkl.Dropout(0.5)(h)
h = tkl.BatchNormalization()(h)
h = tkl.PReLU()(h)
h = tkl.MaxPool2D()(h)

h = tkl.Conv2D(128, 3, padding='same')(h)
h = tkl.BatchNormalization()(h)
h = tkl.PReLU()(h)
h = tkl.MaxPool2D()(h)

h = tkl.Conv2D(128, 3, padding='same')(h)
h = tkl.Dropout(0.5)(h)
h = tkl.BatchNormalization()(h)
h = tkl.PReLU()(h)
h = tkl.MaxPool2D()(h)

h = tkl.Conv2D(256, 3, padding='same')(h)
h = tkl.BatchNormalization()(h)
h = tkl.PReLU()(h)
h = tkl.MaxPool2D()(h)

h = tkl.Conv2D(256, 3, padding='same')(h)
h = tkl.Dropout(0.5)(h)
h = tkl.BatchNormalization()(h)
h = tkl.PReLU()(h)
h = tkl.MaxPool2D()(h)

h = tkl.Conv2D(512, 3, padding='valid')(h)
h = tkl.BatchNormalization()(h)
h = tkl.PReLU()(h)

h = tkl.Flatten()(h)
h = tkl.Dense(512)(h)
h = tkl.PReLU()(h)
y = tkl.Dense(nClusters, activation='softmax')(h)

model = tf.keras.Model(x, y)

model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
lsCallbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                                                patience=5, 
                                                restore_best_weights=True)]
model.fit(x=dictDataTrain['images'],
          y=dictDataTrain['cluster'],
          batch_size=32,
          epochs=20,
          verbose=1,
          callbacks=lsCallbacks,
          validation_data=(dictDataVal['images'], dictDataVal['cluster']))
arrZInferred = model.predict(dictDataUnseen['images'])
