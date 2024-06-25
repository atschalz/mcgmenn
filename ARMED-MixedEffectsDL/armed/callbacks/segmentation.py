'''
Custom Keras callbacks for segmentation models (currently unused)
'''
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class SaveMultiModalImagesCallback(tf.keras.callbacks.Callback):
    def  __init__(self, data, arrYVal, strSaveDir, random_effects=False):
        """Callback for saving some example segmentations as images at the end of each epoch

        Images will contain a 4x4 grid. Each row contains a random validation sample and each 
        column contains a channel/modality. Ground truth masks are overlaid in red while 
        predictions are overlaid in blue.
        Args:
            data (np.array or tuple): 4D array of validation data or tuple of (data, design_matrix)
            arrYVal (np.array): 4D array of validation labels (segmentation masks)
            strSaveDir (str): path to directory for saving images
        """        
        super(SaveMultiModalImagesCallback, self).__init__()
        self.data = data
        self.arrYVal = arrYVal
        self.strSaveDir = strSaveDir
        self.random_effects = random_effects
    def on_epoch_end(self, epoch, logs=None):
        fig, ax = plt.subplots(4, 4, dpi=150)
        # Create a grid where each row is a different sample and each column 
        # is a different modality within that sample
        for i in range(4):
            np.random.seed(i * 2348)
            if self.random_effects:
                k = np.random.randint(self.data[0].shape[0])
                arrInput = self.data[0][k,]
                arrDesign = self.data[1][k,]
                arrPredMask = self.model.predict((np.expand_dims(arrInput, 0),
                                                  np.expand_dims(arrDesign, 0))).squeeze()
            else:
                k = np.random.randint(self.data.shape[0])
                arrInput = self.data[k,] 
                arrPredMask = self.model.predict(np.expand_dims(arrInput, 0)).squeeze()
            arrTrueMask = self.arrYVal[k,].squeeze()
            for j in range(4):
                ax[i, j].imshow(arrInput[:, :, j], cmap='Greys_r')

                arrTrueOverlay = np.zeros(arrTrueMask.shape + (4,))
                arrTrueOverlay[..., 0] = arrTrueMask
                arrTrueOverlay[..., -1] = arrTrueMask
                ax[i, j].imshow(arrTrueOverlay, alpha=0.3)

                arrPredOverlay = np.zeros(arrTrueMask.shape + (4,))
                arrPredOverlay[..., 2] = (arrPredMask >= 0.5)
                arrPredOverlay[..., -1] = (arrPredMask)
                ax[i, j].imshow(arrPredOverlay, alpha=0.3)
                ax[i, j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.strSaveDir, 'epoch{:03d}.png'.format(epoch)))
        plt.close()


class SaveImagesCallback(tf.keras.callbacks.Callback):
    def  __init__(self, data, arrYVal, strSaveDir, random_effects=False):
        """Callback for saving some example segmentations as images at the end of each epoch

        Args:
            data (np.array or tuple): 4D array of validation data or tuple of (data, design_matrix)
            arrYVal (np.array): 4D array of validation labels (segmentation masks)
            strSaveDir (str): path to directory for saving images
        """        
        super(SaveImagesCallback, self).__init__()
        self.data = data
        self.arrYVal = arrYVal
        self.strSaveDir = strSaveDir
        self.random_effects = random_effects
    def on_epoch_end(self, epoch, logs=None):
        fig, ax = plt.subplots(3, 3, dpi=150)
        # Create a grid where each row is a different sample and each column 
        # is a different modality within that sample
        for i in range(9):
            if self.random_effects:
                arrInput = self.data[0][i,]
                arrDesign = self.data[1][i,]
                arrPredMask = self.model.predict((np.expand_dims(arrInput, 0),
                                                  np.expand_dims(arrDesign, 0))).squeeze()
            else:
                arrInput = self.data[i,] 
                arrPredMask = self.model.predict(np.expand_dims(arrInput, 0)).squeeze()
            arrTrueMask = self.arrYVal[i,].squeeze()
            ax.flatten()[i].imshow(arrInput.squeeze(), cmap='Greys_r')

            arrTrueOverlay = np.zeros(arrTrueMask.shape + (4,))
            arrTrueOverlay[..., 0] = arrTrueMask
            arrTrueOverlay[..., -1] = arrTrueMask
            ax.flatten()[i].imshow(arrTrueOverlay, alpha=0.2)

            arrPredOverlay = np.zeros(arrTrueMask.shape + (4,))
            arrPredOverlay[..., 2] = (arrPredMask >= 0.5)
            arrPredOverlay[..., -1] = (arrPredMask)
            ax.flatten()[i].imshow(arrPredOverlay, alpha=0.3)
            ax.flatten()[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.strSaveDir, 'epoch{:03d}.png'.format(epoch)))
        plt.close()