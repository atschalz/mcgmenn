import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def make_spirals(points=1000, classes=2, degrees=360, radius=1, noise=0):
    """Generate the spirals bivariate classification problem

    Args:
        points (int, optional): number of samples. Defaults to 1000.
        classes (int, optional): number of classes. Defaults to 2.
        degrees (int, optional): length of spirals in degrees. Defaults to 360.
        radius (int, optional): radius to each spiral end. Defaults to 1.
        noise (int, optional): added Gaussian noise. Defaults to 0.

    Returns:
        array, array: x, y (one-hot labels if classes > 2)
    """    

    arrNPoints = np.zeros((classes,), dtype=int)
    for iClass in range(classes-1):
        arrNPoints[iClass] = points // classes
    arrNPoints[-1] = points - arrNPoints.sum()
    
    lsX = []
    lsY = []
    
    for iClass in range(classes):
        nPoints = arrNPoints[iClass]
        arrPoints = np.random.randint(0, degrees, size=nPoints) * np.pi / 180
        
        phase = iClass * 2 * np.pi / classes
        
        arrX1 = -np.cos(arrPoints - phase) * arrPoints * radius / arrPoints.max()
        arrX2 = np.sin(arrPoints - phase) * arrPoints * radius / arrPoints.max()
        arrX = np.stack([arrX1, arrX2]).T
        arrX += np.random.normal(scale=noise, size=(nPoints, 2))
        
        arrY = np.zeros((nPoints, classes))
        arrY[:, iClass] = 1
            
        lsX += [arrX]
        lsY += [arrY]
        
    return np.concatenate(lsX, axis=0), np.concatenate(lsY, axis=0)
        

def make_spiral_random_slope(clusters, points_per_cluster=1000, inter_cluster_sd=0.2, 
                             classes=2, degrees=360, radius=1, noise=0):
    """Generate spirals classification problem with data grouped into equal sized 
    clusters. Each cluster has a random effect slope applied to the 2nd feature.

    Args:
        clusters (int): number of clusters
        points_per_cluster (int, optional): Defaults to 1000.
        inter_cluster_sd (float, optional): S.d. of normally distributed random slopes. Defaults to 0.2.
        classes (int, optional): number of classes. Defaults to 2.
        degrees (int, optional): length of spirals in degrees. Defaults to 360.
        radius (int, optional): radius to each spiral end. Defaults to 1.
        noise (int, optional): added Gaussian noise. Defaults to 0.

    Returns:
        features, cluster membership matrix, labels, cluster random slopes
    """    
    
    arrRandomSlopes = np.random.normal(scale=inter_cluster_sd, size=(clusters,))
    
    lsX = []
    lsY = []
    lsZ = []
    for iCluster in range(clusters):
        arrXCluster, arrYCluster = make_spirals(points=points_per_cluster, classes=classes, degrees=degrees,
                                                radius=radius, noise=noise)
        arrXCluster[:, 1] *= (1 / (arrRandomSlopes[iCluster] + 1 + 1e-7))
                
        arrZCluster = np.zeros((arrXCluster.shape[0], clusters))
        arrZCluster[:, iCluster] = 1
        
        lsX += [arrXCluster]
        lsY += [arrYCluster]
        lsZ += [arrZCluster]
        
    return np.concatenate(lsX, axis=0), np.concatenate(lsZ, axis=0), np.concatenate(lsY, axis=0), arrRandomSlopes


def make_spiral_random_radius(clusters, 
                              points_per_cluster=1000, 
                              mean_radius=1.0,
                              inter_cluster_sd=0.2, 
                              classes=2, 
                              degrees=360, 
                              noise=0):
    """Generate spirals classification problem with data grouped into equal sized 
    clusters. Each cluster has a random radius drawn from a normal distribution.

    Args:
        clusters (int): number of clusters
        points_per_cluster (int, optional): Defaults to 1000.
        mean_radius (float, optional): Mean of normally distributed random radii. Defaults to 1.0.
        inter_cluster_sd (float, optional): S.d. of normally distributed random radii. Defaults to 0.2.
        classes (int, optional): number of classes. Defaults to 2.
        degrees (int, optional): length of spirals in degrees. Defaults to 360.
        noise (int, optional): added Gaussian noise. Defaults to 0.

    Returns:
        features, cluster membership matrix, labels, cluster random radii
    """    
    
    arrRadii = np.random.normal(loc=mean_radius, scale=inter_cluster_sd, size=(clusters,))
    
    lsX = []
    lsY = []
    lsZ = []
    for iCluster in range(clusters):
        arrXCluster, arrYCluster = make_spirals(points=points_per_cluster, classes=classes, degrees=degrees,
                                                radius=arrRadii[iCluster], noise=noise)
                        
        arrZCluster = np.zeros((arrXCluster.shape[0], clusters))
        arrZCluster[:, iCluster] = 1
        
        lsX += [arrXCluster]
        lsY += [arrYCluster]
        lsZ += [arrZCluster]
        
    return np.concatenate(lsX, axis=0), np.concatenate(lsZ, axis=0), np.concatenate(lsY, axis=0), arrRadii


def make_spiral_random_radius_confounder(clusters, 
                                         points_per_cluster=1000, 
                                         mean_radius=1.0,
                                         radius_sd=0.2,
                                         ratio_sd=0.2, 
                                         degrees=360, 
                                         noise=0, 
                                         confounders=1):
    """Generate spirals classification problem with data grouped into equal sized 
    clusters. Each cluster has a random radius drawn from a normal distribution. A
    confounding effect is simulated by varying the the class ratio across clusters, 
    then adding one or more confounded independent variables correlated with the
    class ratios.

    Args:
        clusters (int): number of clusters
        points_per_cluster (int, optional): Defaults to 1000.
        mean_radius (float, optional): Mean of normally distributed random radii. Defaults to 1.0.
        radius_sd (float, optional): S.d. of normally distributed random radii. Defaults to 0.2.
        ratio_sd (float, optional): S.d. of normally distributed class ratios, controls 
            strength of confounding effect. Defaults to 0.2. 
        degrees (int, optional): length of spirals in degrees. Defaults to 360.
        noise (int, optional): added Gaussian noise. Defaults to 0.
        confounders (int, optional): number of confounded variables to add. Defaults to 1.

    Returns:
        features, cluster membership matrix, labels, cluster random radii, cluster class ratios
    """ 
    
    arrRadii = np.random.normal(loc=mean_radius, scale=radius_sd, size=(clusters,))
    arrRatio = np.random.normal(loc=0.5, scale=ratio_sd, size=(clusters,))
    arrRatio[arrRatio < 0.1] = 0.1
    arrRatio[arrRatio > 0.9] = 0.9   
    
    lsX = []
    lsY = []
    lsZ = []
    for iCluster in range(clusters):
        arrXCluster, arrYCluster = make_spirals(points=2 * points_per_cluster, degrees=degrees,
                                                radius=arrRadii[iCluster], noise=noise)

        nClass0 = int(points_per_cluster * arrRatio[iCluster])
        nClass1 = points_per_cluster - nClass0

        arrClass0 = np.where(arrYCluster[:, 0])[0]
        arrClass0 = arrClass0[:nClass0]
        arrClass1 = np.where(arrYCluster[:, 1])[0]
        arrClass1 = arrClass1[:nClass1]
        arrPoints = np.concatenate([arrClass0, arrClass1], axis=0)
        
        arrXCluster = arrXCluster[arrPoints]
        arrYCluster = arrYCluster[arrPoints]
        
        arrZCluster = np.zeros((arrXCluster.shape[0], clusters))
        arrZCluster[:, iCluster] = 1
        
        arrConfounders = np.random.normal(loc=arrRatio[iCluster], scale=0.2, size=(points_per_cluster, confounders))
        arrXCluster = np.concatenate((arrXCluster, arrConfounders), axis=1)
        
        lsX += [arrXCluster]
        lsY += [arrYCluster]
        lsZ += [arrZCluster]
        
    return np.concatenate(lsX, axis=0), np.concatenate(lsZ, axis=0), np.concatenate(lsY, axis=0), arrRadii, arrRatio


def make_spiral_true_boundary(classes=2, degrees=360, radius=1):
    """Create points along the "real" decision boundary between 2 spirals

    Args:
        classes (int, optional): number of classes. Defaults to 2.
        degrees (int, optional): length of spirals in degrees. Defaults to 360.
        radius (int, optional): radius to each spiral end. Defaults to 1.
        
    """    
    
    lsX = []
    lsY = []
    
    arrPoints = np.linspace(0, degrees * 1.5, 1000) * np.pi / 180
    radmax = 1.5 * radius / arrPoints.max()
    for iClass in range(classes):
                
        phase = iClass * 2 * np.pi / classes
        
        arrX1 = -np.cos(arrPoints - phase) * arrPoints * radmax
        arrX2 = np.sin(arrPoints - phase) * arrPoints * radmax
        arrX = np.stack([arrX1, arrX2]).T
        
        arrY = np.zeros((1000, classes))
        arrY[:, iClass] = 1
            
        lsX += [arrX]
        lsY += [arrY]
        
    arrSpiralX = np.concatenate(lsX, axis=0)
    arrSpiralY = np.concatenate(lsY, axis=0)
    
    arrSpiral1X = arrSpiralX[arrSpiralY[:, 0] == 1, :]
    arrSpiral2X = arrSpiralX[arrSpiralY[:, 1] == 1, :]

    # For each point in class 1, find nearest point in class 2 and compute the midpoint
    tree = KDTree(arrSpiral2X)
    arrSpiralMid = np.zeros_like(arrSpiral1X)
    for iPoint in range(arrSpiral1X.shape[0]):
        arrPoint1 = arrSpiral1X[iPoint, :]
        _, iPoint2 = tree.query(arrPoint1, 1)
        arrPoint2 = arrSpiral2X[iPoint2, :]
        arrSpiralMid[iPoint] = 0.5 * (arrPoint1 + arrPoint2)

    # Add the inverted points
    arrSpiralMid = np.concatenate([arrSpiralMid, -arrSpiralMid], axis=0)
        
    return arrSpiralMid

        
def plot_clusters(X, Z, Y, random_effects=None, true_spiral_params=None):
    """Plot data points in each cluster.

    Args:
        X (array): independent variables
        Z (array): cluster membership design matrix
        Y (array): labels
        random_effects (array, optional): Cluster-specific random radii, 
            used to create the titles for each subplot. Defaults to None.
        true_spiral_params (dict, optional): Spiral parameters with keys 
            'classes' and 'degrees'. Defaults to None.

    Returns:
        figure, axes
    """    
    nClusters = Z.shape[1]
    nRows = int(np.ceil(nClusters / 5))

    lsMarkers = ['o', 'P', 'X', 'D']    
    fig, ax = plt.subplots(nRows, 5, figsize=(16, 3 * nRows), gridspec_kw={'hspace': 0.5})
    for iCluster in range(nClusters):       
        XCluster = X[Z[:, iCluster] == 1, :]
        YCluster = Y[Z[:, iCluster] == 1,]
         
        iRow = iCluster // 5
        iCol = iCluster % 5

        for iClass in range(Y.shape[1]):
            ax[iRow, iCol].scatter(XCluster[YCluster[:, iClass] == 1, 0], 
                                   XCluster[YCluster[:, iClass] == 1, 1], 
                                   c=f'C{iClass}', 
                                   s=1,
                                   marker=lsMarkers[iClass])
        
        if random_effects is not None:
            ax[iRow, iCol].set_title(f'Random effect: {random_effects[iCluster]:.03f}')
        if iCluster >= 5:
            ax[iRow, iCol].set_xlabel('Feature 1')
        if iCol == 0:
            ax[iRow, iCol].set_ylabel('Feature 2')
            
        if true_spiral_params:
            arrSpiralMid = make_spiral_true_boundary(true_spiral_params['classes'], 
                                                     true_spiral_params['degrees'], 
                                                     random_effects[iCluster])
            
            ax[iRow, iCol].plot(arrSpiralMid[:1000, 0], arrSpiralMid[:1000, 1], c='0.2', ls='--')
            ax[iRow, iCol].plot(arrSpiralMid[1000:, 0], arrSpiralMid[1000:, 1], c='0.2', ls='--')
         
        ax[iRow, iCol].set_xlim(X[:, 0].min(), X[:, 0].max())
        ax[iRow, iCol].set_ylim(X[:, 1].min(), X[:, 1].max())
        ax[iRow, iCol].set_aspect('equal')
        ax[iRow, iCol].set_aspect('equal')
    return fig, ax


def plot_clusters_feature_hist(X, Z, Y, feature_idx, random_effects=None):
    """For each cluster, plot a histogram of feature values..

    Args:
        X (array): independent variables
        Z (array): cluster membership design matrix
        Y (array): labels
        feature_idx (int): feature to plot
        random_effects (array, optional): Cluster-specific random radii, 
            used to create the titles for each subplot. Defaults to None.

    Returns:
        figure, axes
    """    
    
    nClusters = Z.shape[1]
    nRows = int(np.ceil(nClusters / 5))

    lsMarkers = ['o', 'P', 'X', 'D']    
    fig, ax = plt.subplots(nRows, 5, figsize=(16, 3 * nRows), gridspec_kw={'hspace': 0.5})
    for iCluster in range(nClusters):       
        XCluster = X[Z[:, iCluster] == 1, :]
        YCluster = Y[Z[:, iCluster] == 1,]
         
        iRow = iCluster // 5
        iCol = iCluster % 5

        ax[iRow, iCol].hist(XCluster[:, feature_idx], color=(0.5, 0.5, 0.5))
        
        if random_effects is not None:
            ax[iRow, iCol].set_title(f'Class balance: {random_effects[iCluster]:.03f}')
        if iCluster >= 5:
            ax[iRow, iCol].set_xlabel(f'Feature {feature_idx + 1}')
        if iCol == 0:
            ax[iRow, iCol].set_ylabel('Frequency')
         
        ax[iRow, iCol].set_xlim(X[:, feature_idx].min(), X[:, feature_idx].max())

    return fig, ax


def plot_decision_boundary(model, X, Y, Z=None, ax=None, vmax=2):
    """Plot decision boundary learned by a trained model

    Args:
        model (tf.keras.Model): trained model
        X (np.array): input array
        Y (np.array): one-hot labels
        Z (np.array, optional): cluster membership input (design matrix). Defaults to None.
        ax (plt.Axes, optional): matplotlib axes to plot to. Defaults to None.
        vmax (int, optional): how far out in input space to compute decision boundary. Defaults to 2.

    Returns:
        plt.Axes
    """      
    # Create grid of values over X-space     
    arrGridX1, arrGridX2 = np.mgrid[-vmax:vmax+0.1:0.1, -vmax:vmax+0.1:0.1]
    # Assume that the mean of the remaining features is 0.5
    arrGridXConf = np.ones((arrGridX1.size, X.shape[1])) * 0.5
    arrGridX = np.concatenate([arrGridX1.reshape(-1, 1), arrGridX2.reshape(-1, 1), arrGridXConf], axis=1)
    
    if Z is not None:
        arrGridZ = np.zeros((arrGridX.shape[0], Z.shape[1]))
        arrGridZ[:, Z.sum(axis=0).argmax()] = 1
        arrGridYFlat = model.predict((arrGridX, arrGridZ), verbose=0)
    else:
        arrGridYFlat = model.predict(arrGridX, verbose=0) 

    arrGridY = (arrGridYFlat[:, 1] >= 0.5).reshape(arrGridX1.shape).astype(int)
    
    if ax is None:
        ax = plt.gca()
        
    ax.contour(arrGridX1, arrGridX2, arrGridY, levels=1, colors='k')  
    ax.contourf(arrGridX1, arrGridX2, arrGridY, levels=1, colors=['C0', 'C1'], alpha=0.5)    
    ax.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c='C0', s=5, alpha=0.9)
    ax.scatter(X[Y[:, 1] == 1, 0], X[Y[:, 1] == 1, 1], c='C1', s=10, alpha=0.9, marker='P')
        
    return ax