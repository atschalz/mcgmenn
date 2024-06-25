'''
Custom model metrics.
'''
import numpy as np
import sklearn.metrics
import cv2

def compute_youden_point(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute Youden point (where the Youden index sensitivity + 
    specificity - 1 is maximized)

    Args:
        y_true (np.ndarray): true labels
        y_pred (np.ndarray): probabilistic predictions

    Returns:
        float, float: Youden point, maximum Youden index
    """    
    fpr, tpr, thresh = sklearn.metrics.roc_curve(y_true, y_pred)
    youden = tpr - fpr
    youdenPoint = thresh[np.argmax(youden)]
    youdenMax = youden.max()
    
    return youdenPoint, youdenMax

def sensitivity_at_specificity(y_true: np.ndarray, y_pred: np.ndarray, 
                               specificity: float=0.8):
    """Compute sensitivity at fixed specificity.

    Args:
        y_true (np.ndarray): true labels
        y_pred (np.ndarray): probabilistic predictions
        specificity (float, optional): fixed specificity. Defaults to 0.8.

    Returns:
        float: sensitivity
    """    
    from tensorflow.keras.metrics import SensitivityAtSpecificity
    sens = SensitivityAtSpecificity(specificity)(y_true, y_pred)
    return sens.numpy()

def specificity_at_sensitivity(y_true: np.ndarray, y_pred: np.ndarray, 
                               sensitivity: float=0.8):
    """Compute specificity at fixed sensitivity.

    Args:
        y_true (np.ndarray): true labels
        y_pred (np.ndarray): probabilistic predictions
        sensitivity (float, optional): fixed sensitivity. Defaults to 0.8.

    Returns:
        float: specificity
    """ 
    from tensorflow.keras.metrics import SpecificityAtSensitivity
    spec = SpecificityAtSensitivity(sensitivity)(y_true, y_pred)
    return spec.numpy()

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                           youden_point: float=None, 
                           fixed_sens: float=0.8, 
                           fixed_spec: float=0.8):
    """Compute several classification metrics.
    * AUROC
    * At Youden point: balanced accuracy, Youden's index, F1, PPV, NPV, 
        sensitivity, specificity
    * Sensitivity at fixed specificity
    * Specificity at fixed sensitivity

    Args:
        y_true (np.ndarray): true labels
        y_pred (np.ndarray): probabilistic predictions
        youden_point (float, optional): Predetermined Youden point, e.g. based
            on training data. Defaults to None (computes Youden point based on
            y_pred). 
        fixed_sens (float, optional): Compute specificity at this
            sensitivity. Defaults to 0.8. 
        fixed_spec (float, optional): Compute sensitivity at this specificity. 
            Defaults to 0.8.

    Returns:
        dict, float: dictionary of metrics, Youden point

    """    
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    auroc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    y_true = np.array(y_true, dtype=np.bool)

    if youden_point is None:
        youden_point, youden_max = compute_youden_point(y_true, y_pred)
        yPredBinary = y_pred >= youden_point
        fpr = ((1 - y_true) * yPredBinary).sum() / (1 - y_true).sum()
        
    else: 
        yPredBinary = y_pred >= youden_point
        tpr = (y_true * yPredBinary).sum() / y_true.sum()
        fpr = ((1 - y_true) * yPredBinary).sum() / (1 - y_true).sum()
        youden_max = tpr - fpr

    acc = sklearn.metrics.balanced_accuracy_score(y_true, yPredBinary)
    f1 = sklearn.metrics.f1_score(y_true, yPredBinary)
    ppv = sklearn.metrics.precision_score(y_true, yPredBinary)
    npv = sklearn.metrics.precision_score(y_true, yPredBinary, pos_label=0)
    sensitivity = sklearn.metrics.recall_score(y_true, yPredBinary)
    
    sensatspec = sensitivity_at_specificity(y_true, y_pred, specificity=fixed_spec)
    specatsens = specificity_at_sensitivity(y_true, y_pred, sensitivity=fixed_sens)

    return {'AUROC': auroc, 
            'Accuracy': acc, 
            'Youden\'s index': youden_max, 
            'F1': f1, 
            'PPV': ppv, 
            'NPV': npv,
            'Sensitivity at Youden': sensitivity,
            'Specificity at Youden': 1 - fpr,
            f'Sensitivity at {int(fixed_spec * 100)}% Specificity': sensatspec,
            f'Specificity at {int(fixed_sens * 100)}% Sensitivity': specatsens
            }, youden_point

def single_sample_dice(y_true: np.ndarray, y_pred: np.ndarray, threshold: float=0.5):
    """Compute Dice score

    Args:
        y_true (np.ndarray): true label image
        y_pred (np.ndarray): probabilistic predictions
        threshold (float, optional): label threshold. Defaults to 0.5.

    Returns:
        float: Dice score
    """    
    yPredBinary = y_pred >= threshold
    intersection = np.sum(y_true * yPredBinary)
    total = np.sum(y_true) + np.sum(yPredBinary)
    return 2 * intersection / (total + 1e-8)

def balanced_accuracy(y_true, y_pred):
    """Balanced accuracy metric for multi-class labels. Computes the mean 
    of the accuracy of each class.

    Args:
        y_true (tf.Tensor): true labels
        y_pred (tf.Tensor): probabilistic predictions. Will be thresholded 
            at 0.5.

    Returns:
        tf.Tensor: balanced accuracy
    """    
    import tensorflow as tf
    from tensorflow.keras.backend import floatx
    
    predbin = tf.cast((y_pred >= 0.5), floatx())
    correct = tf.cast(tf.equal(y_true, predbin), floatx())
    return tf.reduce_mean(tf.reduce_mean(correct, axis=0))
    
def image_metrics(img: np.ndarray):
    '''Compute image metrics, including brightness, contrast, sharpness, and
    SNR
    
    Args:
        img (np.ndarray): image
        
    Returns:
        dict: metrics
    
    '''
    
    brightness = img.mean()
    contrast = img.std()
    sharpness = cv2.Laplacian(img, cv2.CV_32F).var()
    snr = brightness/contrast
    return {'Brightness': brightness,
            'Contrast': contrast,
            'Sharpness': sharpness,
            'SNR': snr}
