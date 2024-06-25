'''
Select images for inclusion in analysis. These will include cells from all 7
PDXs and from any date (batch) where at least 2 PDXs were imaged. Images from
heldout dates (where < 2 PDXs were imaged) will be reserved for testing model
generalization to "unseen batches". 

Each single cell was imaged across hundreds of timepoints. Since adjacent
timepoints will look very similar, we will select every 4th frame for more
efficient model training. For the heldout dates, we will select every 32nd
frame.

Then, divide dataset into 70% train, 10% validation, and 20% test. Save selected
images and associated label and batch information into .npz files. Metadata
tables containing original image paths and cell information are saved as .csv
files.
'''

import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from armed.misc import expand_data_path
from armed.crossvalidation.grouped_cv import StratifiedGroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from PIL import Image

# Path to .png images 
strSourceDir = expand_data_path('melanoma_hackathon/raw/png')
# Where to save dataset
strOutputDir = expand_data_path('melanoma/allpdx_selecteddates', make=True)

# Data source directory has structure:
# .../png/<metastatic efficiency>/<pdx name>/<date>_<pdxs>/<cell name>/<frame>.png

# List date directories
lsDateDirsHigh = glob.glob(os.path.join(strSourceDir, 'high', '*', '*'))
lsDateDirsLow = glob.glob(os.path.join(strSourceDir, 'low', '*', '*'))
lsDateDirs = lsDateDirsHigh + lsDateDirsLow

# Parse dates from 
def _get_date(x):
    base = x.split(os.path.sep)[-1]
    return base.split('_')[0]
lsDates = list(map(_get_date, lsDateDirs))
# Count PDXs imaged on each date
arrDates, arrPDXsPerDay = np.unique(lsDates, return_counts=True)

# Select all days with at least 2 PDX's
arrDatesIncluded = arrDates[arrPDXsPerDay >= 2]
# Reserve remaining days for "unseen" batches dataset
arrDatesHeldout = arrDates[arrPDXsPerDay < 2]

print(f'Including {len(arrDatesIncluded)} dates:', arrDatesIncluded)
print(f'Holding out {len(arrDatesHeldout)} dates:', arrDatesHeldout)

# Count number of cells per included date
lsCellDirs = []
for date in arrDatesIncluded:
    lsCellDirs += glob.glob(os.path.join(strSourceDir, 'high', '*', date + '*', '*'))
    lsCellDirs += glob.glob(os.path.join(strSourceDir, 'low', '*', date + '*', '*'))
    
print(f'{len(lsCellDirs)} cells imaged across all included dates')

# From each cell, pick every 4th frame
lsImages = []
for strCellDir in lsCellDirs:
    lsCellImages = glob.glob(os.path.join(strCellDir, '*.png'))
    lsImages += lsCellImages[::4]
    
print(f'{len(lsImages)} images')

# Construct a dataframe with image metadata:
#   image path
#   date
#   PDX (cell type)
#   metastatic efficiency
#   cell name
def make_dataframe(images):
    dfImageInfo = pd.DataFrame(columns=['image', 'date', 'celltype', 'met-eff', 'cell'], 
                            index=range(len(images)))
    for i, strImagePath in enumerate(images):
        strFileName = os.path.basename(strImagePath)
        strDate, strCelltype = strFileName.split('_')[:2]
        strMetEff = 'high' if 'high' in strImagePath else 'low'
        strCell = strImagePath.split(os.path.sep)[-2]
        dfImageInfo.loc[i] = [strImagePath, strDate, strCelltype, strMetEff, strCell]
    return dfImageInfo

dfInfo = make_dataframe(lsImages)

# Do the same for the heldout days
lsCellDirs = []
for date in arrDatesHeldout:
    lsCellDirs += glob.glob(os.path.join(strSourceDir, 'high', '*', date + '*', '*'))
    lsCellDirs += glob.glob(os.path.join(strSourceDir, 'low', '*', date + '*', '*'))
    
print(f'{len(lsCellDirs)} cells imaged across all excluded dates')

# From each cell, pick every 32nd frame (we don't need that many for evaluation)
lsImagesHeldout = []
for strCellDir in lsCellDirs:
    lsCellImages = glob.glob(os.path.join(strCellDir, '*.png'))
    lsImagesHeldout += lsCellImages[::32]
    
print(f'{len(lsImagesHeldout)} images')

dfInfoHeldout = make_dataframe(lsImagesHeldout)

# Stratify splits so that each partition has similar representation of PDX
# and acquisition dates. To use StratifiedGroupSHuffleSplit with 2 levels of
# stratification, we create a temporary "label" consisting of the PDX name
# concatenated to the date.
dfStrat = dfInfo.apply(lambda row: '_'.join([row['celltype'], row['date']]), axis=1)
# Split into 70% train, 10% validation, 20% test
splitTest = StratifiedGroupShuffleSplit(test_size=0.2, n_splits=1, random_state=38)
arrIdxTrainVal, arrIdxTest = next(splitTest.split(dfInfo, 
                                                  dfStrat.values.astype(str), 
                                                  groups=dfInfo['cell'].values.astype(str)))
dfTrainVal = dfInfo.iloc[arrIdxTrainVal]
dfTest = dfInfo.iloc[arrIdxTest]

splitVal = StratifiedGroupShuffleSplit(test_size=0.125, n_splits=1, random_state=38) # 0.125 x 0.8 = 0.1
arrIdxTrain, arrIdxVal = next(splitVal.split(dfTrainVal, 
                                             dfStrat.iloc[arrIdxTrainVal].values.astype(str), 
                                             groups=dfTrainVal['cell'].values.astype(str)))
dfTrain = dfTrainVal.iloc[arrIdxTrain]
dfVal = dfTrainVal.iloc[arrIdxVal]

# Save out metadata
dfTrain.to_csv(os.path.join(strOutputDir, 'data_train.csv'))
dfVal.to_csv(os.path.join(strOutputDir, 'data_val.csv'))
dfTest.to_csv(os.path.join(strOutputDir, 'data_test.csv'))
dfInfoHeldout.to_csv(os.path.join(strOutputDir, 'data_unseen.csv'))

print(dfTrain.shape[0], 'train images')
print(dfVal.shape[0], 'val images')
print(dfTest.shape[0], 'test images')
print(dfInfoHeldout.shape[0], 'heldout day images')

# Load .png images, convert to numpy arrays, and scale intensity range to [0, 1]
def images_to_numpy(paths):
    n = len(paths)
    arrImages = np.zeros((n, 256, 256, 1), dtype=np.float32)
    for i, p in tqdm(enumerate(paths), total=n):
        img = Image.open(p)
        arrImages[i,:, :, 0] = np.array(img) / 255.
    return arrImages

arrImagesTrain = images_to_numpy(dfTrain['image'])
arrImagesVal = images_to_numpy(dfVal['image'])
arrImagesTest = images_to_numpy(dfTest['image'])
arrImagesHeldout = images_to_numpy(dfInfoHeldout['image'])

# Create cluster-membership design matrix through one-hot encoding
onehot = OneHotEncoder(sparse=False, dtype=np.float32)
arrClusterTrain = onehot.fit_transform(dfTrain['date'].values.reshape((-1, 1)))
arrClusterVal = onehot.transform(dfVal['date'].values.reshape((-1, 1)))
arrClusterTest = onehot.transform(dfTest['date'].values.reshape((-1, 1)))
arrClusterHeldout = np.zeros((arrImagesHeldout.shape[0], arrClusterTrain.shape[1]), dtype=np.float32)

# Save mapping of one-hot columns to cluster names
arrClusterOrder = onehot.categories_[0]
np.savetxt(os.path.join(strOutputDir, 'cluster_order.txt'), arrClusterOrder, fmt='%s')

# Create binary labels
arrLabelTrain = (dfTrain['met-eff'].values == 'low').astype(np.float32)
arrLabelVal = (dfVal['met-eff'].values == 'low').astype(np.float32)
arrLabelTest = (dfTest['met-eff'].values == 'low').astype(np.float32)
arrLabelHeldout = (dfInfoHeldout['met-eff'].values == 'low').astype(np.float32)

# Save out datasets as .npz files
np.savez_compressed(os.path.join(strOutputDir, 'data_train.npz'),
                    images=arrImagesTrain,
                    label=arrLabelTrain,
                    cluster=arrClusterTrain)
np.savez_compressed(os.path.join(strOutputDir, 'data_val.npz'),
                    images=arrImagesVal,
                    label=arrLabelVal,
                    cluster=arrClusterVal)
np.savez_compressed(os.path.join(strOutputDir, 'data_test.npz'),
                    images=arrImagesTest,
                    label=arrLabelTest,
                    cluster=arrClusterTest)
np.savez_compressed(os.path.join(strOutputDir, 'data_unseen.npz'),
                    images=arrImagesHeldout,
                    label=arrLabelHeldout,
                    cluster=arrClusterHeldout)