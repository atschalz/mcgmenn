'''
Meta-learning wrapper function to train a conventional model to be
domain-agnostic.
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.python.platform.tf_logging import flush

def _get_batches(cluster_batches_dict, clusters, training_iter):
    # Get the next batch from each cluster
    lsBatchesX = []
    lsBatchesY = []
    for iCluster in clusters:
        # Check if batches have been exhausted for this cluster
        if training_iter < (len(cluster_batches_dict[iCluster]) - 1):
            lsBatchesX += [cluster_batches_dict[iCluster][training_iter]['x']]
            lsBatchesY += [cluster_batches_dict[iCluster][training_iter]['y']]
        else:
            # Pick a random batch
            r = np.random.randint(0, len(cluster_batches_dict[iCluster]))
            lsBatchesX += [cluster_batches_dict[iCluster][r]['x']]
            lsBatchesY += [cluster_batches_dict[iCluster][r]['y']]
    return np.concatenate(lsBatchesX, axis=0), np.concatenate(lsBatchesY, axis=0)
        
        
def mldg(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
         model: tf.keras.Model, 
         outer_lr: float=0.001,
         inner_lr: float=0.001,
         epochs: int=40,
         cluster_batch_size: int=4,
         loss_fn=categorical_crossentropy,
         meta_test_weight: float=1.,
         verbose: bool=False):
    """Implementation of metalearning domain generalization by Li 2018
    https://arxiv.org/abs/1710.03463
    
    Trains a conventional model using MLDG.

    Args:
        X (np.ndarray): data
        Y (np.ndarray): labels
        Z (np.ndarray): one-hot encoded cluster (domain) membership
        model (tf.keras.Model): model
        outer_lr (float, optional): Outer learning rate. Defaults to 0.001.
        inner_lr (float, optional): Inner learning rate. Defaults to 0.001.
        epochs (int, optional): Training duration. Defaults to 40.
        cluster_batch_size (int, optional): Per-domain batch size. Defaults to 4.
        loss_fn (optional): Loss function. Defaults to categorical_crossentropy.
        meta_test_weight (float, optional): Weighting of meta-test gradient step 
            relative to meta-training gradient step. Defaults to 1.
        verbose (bool, optional): Defaults to False.

    Returns:
        tf.keras.Model: trained model
    """        
    # Create TF datasets for ease of minibatching
    nClusters = Z.shape[1]
    dictData = {}
    for i in range(nClusters):
        arrXCluster = X[Z[:, i] == 1, :]
        arrYCluster = Y[Z[:, i] == 1,]
        arrZCluster = Z[Z[:, i] == 1, :]
        
        dictData[i] = tf.data.Dataset.from_tensor_slices({'x': arrXCluster, 'y': arrYCluster, 'z': arrZCluster})
    
    opt = tf.keras.optimizers.Adam(learning_rate=outer_lr)
    optInner = tf.keras.optimizers.SGD(learning_rate=inner_lr)
    
    for iEpoch in range(epochs):
        # Reshuffle minibatches
        dictMiniBatches = {k: list(v.shuffle(1000).batch(cluster_batch_size).as_numpy_iterator()) \
            for k, v in dictData.items()}
        
        # Compute max possible batches per cluster
        nIter = np.max([len(x) for x in dictMiniBatches.values()])
        for iIter in range(nIter): 
            # Pick a cluster to be meta-test
            arrMetaTestClusters = np.random.choice(np.arange(nClusters), size=1)       
            # The rest are assigned to meta-train
            arrMetaTrainClusters = np.array([x for x in np.arange(nClusters) if x not in arrMetaTestClusters])
            
            # Get batch data from meta-train and meta-test
            arrMetaTrainX, arrMetaTrainY = _get_batches(dictMiniBatches, arrMetaTrainClusters, iIter)
            arrMetaTestX, arrMetaTestY = _get_batches(dictMiniBatches, arrMetaTestClusters, iIter)

            with tf.GradientTape() as gt1:
                lsWeightsOld = model.get_weights()
                gt1.watch(model.trainable_weights)
            
                # Gradient descent step using meta-train data
                with tf.GradientTape() as gt2:
                    predMetaTrain = model(arrMetaTrainX.astype(np.float32))
                    lossMetaTrain = loss_fn(arrMetaTrainY.astype(np.float32), predMetaTrain)
                    lossMetaTrain = tf.reduce_sum(lossMetaTrain) / arrMetaTrainX.shape[0]
        
                    lsGradsMetaTrain = gt2.gradient(lossMetaTrain, model.trainable_weights)
                    optInner.apply_gradients(zip(lsGradsMetaTrain, model.trainable_weights))

                # Compute loss on meta-test using new weights
                predMetaTest = model(arrMetaTestX.astype(np.float32))
                lossMetaTest = loss_fn(arrMetaTestY.astype(np.float32), predMetaTest)
                lossMetaTest = tf.reduce_sum(lossMetaTest) / arrMetaTestX.shape[0]
            
                model.set_weights(lsWeightsOld)
            
            # Compute gradient of meta-test loss wrt original weights
            lsGradsMetaTest = gt1.gradient(lossMetaTest, model.trainable_weights)
            
            # Update weights using weighted combination of meta-train and meta-test gradients
            lsGradsTotal = [x * meta_test_weight + y for x, y in zip(lsGradsMetaTest, lsGradsMetaTrain)]
            opt.apply_gradients(zip(lsGradsTotal, model.trainable_weights))      
            
        if verbose:
            acc = model.evaluate(X, Y, verbose=0)
            print(f'{iEpoch}/{epochs} - {acc}', flush=True)
        
    return model