'''
Custom callbacks for autoencoder-classifiers.
'''

import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

from armed.metrics import image_metrics
from scipy.stats import f_oneway

def make_recon_figure_callback(images: np.array, 
                               model, 
                               output_dir: str, 
                               clusters: np.array=None, 
                               mixedeffects: bool=False):
    """Generate a callback function that produces a figure with example 
    reconstructions. The figure optionally includes the reconstructions 
    with and without cluster-specific effects. The generated function 
    should be used with the LambdaCallback class from Keras to create
    the callback object.

    Args:
        images (np.array): batch of 8 images (8 x h x w x 1)
        model (tf.keras.Model): model        
        output_dir (str): output path
        clusters (np.array): one-hot encoded cluster design matrix if 
            needed by model (8 x n_clusters). Defaults to None
        mixedeffects (bool): include recons w/ and w/o random effects
    """    
    
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    if mixedeffects:
        
        def _recon_images(epoch, logs):
            # Callback function for saving example reconstruction images after each epoch
            fig, ax = plt.subplots(4, 9, figsize=(9, 4),
                                gridspec_kw={'hspace': 0.3, 'width_ratios': [1] * 8 + [0.2]})  
        
            arrReconME, arrReconFE = model.predict((images, clusters))[:2]
            arrReconDiff = arrReconME - arrReconFE
            vmax = np.abs(arrReconDiff).max()

            for iImg in range(8):
                ax[0, iImg].imshow(images[iImg,], cmap='gray', vmin=0., vmax=1.)
                ax[1, iImg].imshow(arrReconFE[iImg,], cmap='gray', vmin=0., vmax=1.)
                ax[2, iImg].imshow(arrReconME[iImg,], cmap='gray', vmin=0., vmax=1.)
                ax[3, iImg].imshow(arrReconDiff[iImg,], cmap='coolwarm', vmin=-vmax, vmax=vmax)
                
                ax[0, iImg].axis('off')
                ax[1, iImg].axis('off')
                ax[2, iImg].axis('off')
                ax[3, iImg].axis('off')
            
            ax[0, 0].text(-0.2, 0.5, 'Original', transform=ax[0, 0].transAxes, va='center', ha='center', rotation=90)    
            ax[1, 0].text(-0.2, 0.5, 'Recon: FE', transform=ax[1, 0].transAxes, va='center', ha='center', rotation=90)
            ax[2, 0].text(-0.2, 0.5, 'Recon: ME', transform=ax[2, 0].transAxes, va='center', ha='center', rotation=90)    
            ax[3, 0].text(-0.2, 0.5, '(ME - FE)', transform=ax[3, 0].transAxes, va='center', ha='center', rotation=90)
            
            for a in ax[:, -1]:
                a.remove()
            
            axCbar = fig.add_subplot(ax[0, -1].get_gridspec()[:, -1])
            fig.colorbar(ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax),
                                        cmap='coolwarm'),
                        cax=axCbar)
            axCbar.set_ylabel('Difference (ME - FE)')
            
            fig.tight_layout(w_pad=0.1, h_pad=0.1)
            fig.savefig(os.path.join(output_dir, f'epoch{epoch+1:03d}.png'))
            plt.close(fig)
    
    else:
        def _recon_images(epoch, logs):
            # Callback function for saving example reconstruction images after each epoch
            fig, ax = plt.subplots(2, 8, figsize=(8, 2))  
        
            if clusters is not None:
                arrRecon = model.predict((images, clusters))[0]
            else:
                arrRecon = model.predict(images)[0]
            
            for iImg in range(8):
                ax[0, iImg].imshow(images[iImg,], cmap='gray', vmin=0., vmax=1.)
                ax[1, iImg].imshow(arrRecon[iImg,], cmap='gray', vmin=0., vmax=1.)
                
                ax[0, iImg].axis('off')
                ax[1, iImg].axis('off')
            
            ax[0, 0].text(-0.2, 0.5, 'Original', transform=ax[0, 0].transAxes, va='center', ha='center', rotation=90)    
            ax[1, 0].text(-0.2, 0.5, 'Recon', transform=ax[1, 0].transAxes, va='center', ha='center', rotation=90)
            
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=UserWarning)  
                fig.tight_layout(w_pad=0.1, h_pad=0.1)
            fig.savefig(os.path.join(output_dir, f'epoch{epoch+1:03d}.png'))
            plt.close(fig)
            
    return _recon_images

def make_compute_latents_callback(model, images: np.array, image_metadata: pd.DataFrame, output_dir: str):
    """Generate a callback function that calls the encoder on some images
    to create latent representations, then saves them to a .pkl file. The
    function also computes the Davies-Bouldin and Calinski-Harabasz clustering
    metrics on the latents and logs the results to a file. The generated
    function should be used with the LambdaCallback class from Keras to create
    the callback object.

    Args:
        model (tf.keras.Model): encoder model
        images (np.array): batch of 8 images (8 x h x w x 1)
        image_metadata (pd.DataFrame): metadata table
        output_dir (str): output path
    """ 

    def _compute_latents(epoch, logs):
        # callback function for computing latent reps for all training images and saving to a pkl file
        arrLatents = model.predict(images)
        dfLatents = pd.DataFrame(arrLatents, index=image_metadata['image'].values)
        dfLatents.to_pickle(os.path.join(output_dir, f'epoch{epoch+1:03d}_latents.pkl'))
        
        db = davies_bouldin_score(dfLatents, image_metadata['date'])
        ch = calinski_harabasz_score(dfLatents, image_metadata['date'])

        print(f'\nClustering scores:'
            f'\n\tDavies-Bouldin (higher is better): {db}'
            f'\n\tCalinski-Harabasz (lower is better): {ch}'
        )
        
        # Append to file
        with open(os.path.join(output_dir, 'clustering_scores.csv'), 'a') as f:
            if epoch == 0:
                f.write('epoch,DB,CH\n')
            f.write(f'{epoch+1},{db},{ch}\n')
            
        
    return _compute_latents
    
def compute_image_metrics(epoch: int, model, data_in, metadata: pd.DataFrame, 
                          output_dir: str, output_idx: int=0):
    """Compute image metrics including brightness, contrast, sharpness, and SNR. 
    Also create histograms comparing distributions of these metrics across clusters.

    Args:
        epoch (int): epoch number
        model (tf.keras.Model): model
        data_in (np.array or tuple of arrays): input data
        metadata (pd.DataFrame): image metadata
        output_dir (str): path to output location
        output_idx (int, optional): Index of model outputs containing the image 
            outputs. Defaults to 0.

    Returns:
        [type]: [description]
    """    
    
    lsRecons = []
    if isinstance(data_in, tuple):
        nImages = data_in[0].shape[0]
    else:
        nImages = data_in.shape[0]
    nBatches = int(np.ceil(nImages / 1000))

    for iBatch in range(nBatches):
        iStart = 1000 * iBatch
        iEnd = np.min([1000 * (iBatch + 1), nImages])
        
        if isinstance(data_in, tuple):
            batch_in = (data_in[0][iStart:iEnd,], data_in[1][iStart:iEnd,])
        else:
            batch_in = data_in[iStart:iEnd,]
            
        arrRecons = model.predict(batch_in, batch_size=32)[output_idx]        
        lsRecons += [arrRecons]
        
    arrRecons = np.concatenate(lsRecons, axis=0)

    lsMetrics = [image_metrics(img) for img in arrRecons]

    dfMetrics = pd.DataFrame(lsMetrics)
    dfMetrics.index = metadata.index

    # dictDates = {160802: 'Day 1',
    #             160808: 'Day 2',
    #             161209: 'Day 3',
    #             161214: 'Day 4',
    #             161220: 'Day 5',
    #             161224: 'Day 6'}
    
    # dfMetrics['Date'] = metadata['date'].apply(lambda x: dictDates[x]).values
    dfMetrics['Date'] = metadata['date']
    
    dictMetricNames = {'Brightness': 'Mean brightness',
                       'Contrast': 'Contrast (s.d.)',
                       'Sharpness': 'Sharpness (variance-of-Laplacian)',
                       'SNR': 'Signal-to-noise ratio'}

    dictFstats = {}
    fig, ax = plt.subplots(4, 1, figsize=(16, 13), gridspec_kw={'hspace': 0.4})
    for i, (strMetric, strAxisLabel) in enumerate(dictMetricNames.items()):
        vmax = dfMetrics[strMetric].quantile(0.999)
        vmin = dfMetrics[strMetric].min()
        sns.histplot(data=dfMetrics[(dfMetrics[strMetric] >= vmin) & (dfMetrics[strMetric] <= vmax)], 
                     x=strMetric, hue='Date', ax=ax[i], stat='density', bins=100)
        ax[i].set_xlabel(strAxisLabel)
        
        lsGroups = [dfMetrics[strMetric].loc[dfMetrics['Date'] == d].values for d in dfMetrics['Date'].unique()]
        f, p = f_oneway(*lsGroups)
        dictFstats[strMetric] = f
        
    fig.savefig(os.path.join(output_dir, f'epoch{epoch:03d}_recon_image_metrics.svg'))
    plt.close(fig)
    return dictFstats
    
def make_image_metrics_callback(model, data_in, metadata, output_dir, output_idx=0):
    """Generate a callback function that computes image metrics including 
    brightness, contrast, sharpness, and SNR. The generated function should be
    used with the LambdaCallback class from Keras to create the callback object.
    
    Args:
        model (tf.keras.Model): model
        data_in (np.array or tuple of arrays): input data
        metadata (pd.DataFrame): image metadata
        output_dir (str): path to output location
        output_idx (int, optional): Index of model outputs containing the image 
            outputs. Defaults to 0.
    """    
    def _fn(epoch , logs):
        metrics = compute_image_metrics(epoch+1, model, data_in, metadata, output_dir, output_idx=output_idx)
        print(metrics)

        # Append to file
        metrics['Epoch'] = epoch + 1
        lsKeys = ['Epoch', 'Brightness', 'Contrast', 'Sharpness', 'SNR']
        with open(os.path.join(output_dir, 'image_metrics_fstat.csv'), 'a') as f:
            if epoch == 0:
                f.write(','.join(lsKeys) + '\n')
            f.write(','.join([str(metrics[k]) for k in lsKeys]) + '\n')

    return _fn