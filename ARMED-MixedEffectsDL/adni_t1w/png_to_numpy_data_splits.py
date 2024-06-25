'''
Convert images into a numpy array, then divide into 70% train/10% val/20% test partitions.
'''

import os
import glob
import re
from PIL import Image
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from armed.misc import expand_data_path

strImageDir = expand_data_path('ADNI23_sMRI/right_hippocampus_slices_2pctnorm/coronal_MNI-6_qc/good')
strOutDir = expand_data_path('ADNI23_sMRI/right_hippocampus_slices_2pctnorm/coronal_MNI-6_numpy/12sites', 
                             make=True)
# 5 sites with GE scanners and majority AD, 7 sites with Philips/Siemens scanners and majority CN
lsSitesKept = [52, 5, 126, 57, 16, 2, 73, 100, 41, 22, 941, 20]
# Number of Monte Carlo random splits to generate
nSplits = 10 

dfImageInfo = pd.read_csv('image_list_ad_cn.csv', index_col=0)
dfImageInfo['RID'] = dfImageInfo['RID'].apply(lambda x: f'{int(x):04d}')
dfImageInfo['ScanDate'] = dfImageInfo['ScanDate'].apply(lambda x: x.replace('-', ''))
dfImageInfo.index = pd.MultiIndex.from_frame(dfImageInfo[['RID', 'ScanDate']])

lsImages = glob.glob(os.path.join(strImageDir, '*'))
lsImages.sort()

nImages = len(lsImages)
arrImages = np.zeros((nImages, 192, 192, 1), dtype=np.float32)
lsKeptImages = []

for i, strImagePath in enumerate(lsImages):
    match = re.search(r'sub-(\d+)_ses-(\d+)', strImagePath)
    strSub = match[1]
    strSes = match[2]
    lsKeptImages += [(strSub, strSes)]
    
    img = np.array(Image.open(strImagePath))
    arrImages[i, :, :, 0] = img / 255.
    
dfImageInfoKept = dfImageInfo.loc[lsKeptImages]
arrImagesIncludedSites = arrImages[dfImageInfoKept['Site'].isin(lsSitesKept),]
dfImageInfoIncludedSites = dfImageInfoKept.loc[dfImageInfoKept['Site'].isin(lsSitesKept)]
arrImagesExcludedSites = arrImages[~dfImageInfoKept['Site'].isin(lsSitesKept),]
dfImageInfoExcludedSites = dfImageInfoKept.loc[~dfImageInfoKept['Site'].isin(lsSitesKept)]
dfImageInfoExcludedSites.to_csv(os.path.join(strOutDir, 'data_unseen.csv'))

arrSites = dfImageInfoIncludedSites['Site'].values
arrLabels = (dfImageInfoIncludedSites['DX_Scan'].values == 'Dementia').astype(np.float32)
    
# Create one-hot design matrix encoding cluster membership
onehot = OneHotEncoder(sparse=False)
arrClusters = onehot.fit_transform(arrSites.reshape((-1, 1))).astype(np.float32)
arrSiteOrder = onehot.categories_[0]

# Split the subjects, stratifying by site and diagnosis. Ignore the number of
# images per subject, since there isn't enough data to stratify to that level of
# detail.
dfSubID = dfImageInfoIncludedSites.index.get_level_values(0)
arrSubID = dfSubID[~dfSubID.duplicated()].values
dfStrat = dfImageInfoIncludedSites.apply(lambda x: str(x['Site']) + '_' + x['DX_Scan'], axis=1)
arrStrat = dfStrat.loc[~dfSubID.duplicated()].values
arrSubID = arrSubID.astype(str)
arrStrat = arrStrat.astype(str)

# Check for subjects who switched diagnoses during the study
lsBadSubs = []
for sub in arrSubID:
    dfLabelsSub = dfImageInfoIncludedSites['DX_Scan'].loc[sub]
    if np.any(dfLabelsSub != dfLabelsSub.iloc[0]):
        print(sub, 'switches diagnosis:', dfLabelsSub.values)
        lsBadSubs += [sub]      
arrStrat = arrStrat[~np.isin(arrSubID, lsBadSubs)]
arrSubID = arrSubID[~np.isin(arrSubID, lsBadSubs)]

# Ignore any site-diagnosis combinations that have only 1 subject
lsBadStrats = []
for strat in np.unique(arrStrat):
    arrSubsStrat = arrSubID[arrStrat == strat,]
    if len(np.unique(arrSubsStrat)) < 2:
        print(strat, 'only has one subject')
        lsBadStrats += [strat]
arrSubID = arrSubID[~np.isin(arrStrat, lsBadStrats)]
arrStrat = arrStrat[~np.isin(arrStrat, lsBadStrats)]

testsplit = StratifiedShuffleSplit(n_splits=nSplits, test_size=0.2, random_state=32)
for iSplit, (arrTrainValIdx, arrTestIdx) in enumerate(testsplit.split(arrSubID, arrStrat)):
      print('===== Split', iSplit, '=====')
      arrSubIDTrainVal = arrSubID[arrTrainValIdx,]
      arrSubIDTest = arrSubID[arrTestIdx,]
      arrStratTrainVal = arrStrat[arrTrainValIdx,]

      arrImagesTest = arrImagesIncludedSites[dfSubID.isin(arrSubIDTest),]
      arrLabelsTest = arrLabels[dfSubID.isin(arrSubIDTest),]
      arrClustersTest = arrClusters[dfSubID.isin(arrSubIDTest),]
      arrImagesTrainVal = arrImagesIncludedSites[dfSubID.isin(arrSubIDTrainVal),]
      arrLabelsTrainVal = arrLabels[dfSubID.isin(arrSubIDTrainVal),]
      arrClustersTrainVal = arrClusters[dfSubID.isin(arrSubIDTrainVal),]

      print('Test set:', len(arrSubIDTest), 'subjects,', arrImagesTest.shape[0], 'images',
            f'{arrLabelsTest.mean()*100:.03f}% AD')

      valsplit = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=32)
      arrTrainIdx, arrValIdx = next(testsplit.split(arrSubIDTrainVal, arrStratTrainVal))
      arrSubIDTrain = arrSubIDTrainVal[arrTrainIdx,]
      arrSubIDVal = arrSubIDTrainVal[arrValIdx,]

      arrImagesVal = arrImagesIncludedSites[dfSubID.isin(arrSubIDVal),]
      arrLabelsVal = arrLabels[dfSubID.isin(arrSubIDVal),]
      arrClustersVal = arrClusters[dfSubID.isin(arrSubIDVal),]
      arrImagesTrain = arrImagesIncludedSites[dfSubID.isin(arrSubIDTrain),]
      arrLabelsTrain = arrLabels[dfSubID.isin(arrSubIDTrain),]
      arrClustersTrain = arrClusters[dfSubID.isin(arrSubIDTrain),]

      print('Val set:', len(arrSubIDVal), 'subjects,', arrImagesVal.shape[0], 'images',
            f'{arrLabelsVal.mean()*100:.03f}% AD')
      print('Train set:', len(arrSubIDTrain), 'subjects,', arrImagesTrain.shape[0], 'images',
            f'{arrLabelsTrain.mean()*100:.03f}% AD')

      # Unseen sites 
      arrLabelsUnseen = (dfImageInfoExcludedSites['DX_Scan'].values == 'Dementia').astype(np.float32)
      arrClustersUnseen = np.zeros((arrImagesExcludedSites.shape[0], arrSiteOrder.shape[0]), dtype=np.float32)
      print('Unseen sites:', len(dfImageInfoExcludedSites['RID'].unique()), 'subjects,', 
            arrImagesExcludedSites.shape[0], 'images',
            f'{arrLabelsUnseen.mean()*100:.03f}% AD')
      
      strSplitDir = os.path.join(strOutDir, f'split{iSplit:02d}')
      os.makedirs(strSplitDir, exist_ok=True)
      
      np.savez(os.path.join(strSplitDir, 'data_test.npz'), images=arrImagesTest, label=arrLabelsTest, 
            cluster=arrClustersTest, siteorder=arrSiteOrder)
      np.savez(os.path.join(strSplitDir, 'data_val.npz'), images=arrImagesVal, label=arrLabelsVal, 
            cluster=arrClustersVal, siteorder=arrSiteOrder)
      np.savez(os.path.join(strSplitDir, 'data_train.npz'), images=arrImagesTrain, label=arrLabelsTrain, 
            cluster=arrClustersTrain, siteorder=arrSiteOrder)
      np.savez(os.path.join(strSplitDir, 'data_unseen.npz'), images=arrImagesExcludedSites,
            label=arrLabelsUnseen, cluster=arrClustersUnseen, siteorder=arrSiteOrder)