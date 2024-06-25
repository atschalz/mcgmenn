'''
Get key information and measurements from each preprocessed T1w MRI. Save to a .csv table.

Values obtained:
1. Scanner manufacturer
2. Scanner model
3. Echo time, repetition time
4. Flip angle
5. Voxel size (resolution)
6. Mean R hippocampus intensity
7. Mean and S.D. brain intensity
8. Hippocampus edge contrasts
'''
import os
import re
import glob
import json
import nibabel
import pandas as pd
import numpy as np
from scipy.ndimage import morphology
import tqdm
import nilearn.image
from nilearn.datasets import fetch_atlas_aal
from armed.misc import expand_data_path

# Preprocessed image root
strDataPath = expand_data_path('ADNI23_sMRI/DLLabPipeline_sMRI_20220103')

# Load image
dfInfo = pd.read_csv('image_list_ad_cn.csv', index_col=0)
dfInfo['RID'] = dfInfo['RID'].apply(lambda x: f'{int(x):04d}')
dfInfo['ScanDate'] = dfInfo['ScanDate'].apply(lambda x: x.replace('-', ''))
dfInfo.index = pd.MultiIndex.from_frame(dfInfo[['RID', 'ScanDate']])

# Get list of image directories
lsImageDirs = glob.glob(os.path.join(strDataPath, 'sub*', 'ses*', 'anat'))
lsImageDirs.sort()

dictAtlas = fetch_atlas_aal()
iHippocampus = 4102
imgAtlas = nibabel.load(dictAtlas['maps'])
imgHippo = nilearn.image.math_img(f'img == {iHippocampus}', img=imgAtlas)


lsMetrics = []
for strImageDir in tqdm.tqdm(lsImageDirs):
    regmatch = re.search(r'sub-(\d+).*ses-(\d+)', strImageDir)
    strSub = regmatch[1]
    strSes = regmatch[2]
    strSite = dfInfo.loc[(strSub, strSes), 'Site']
    strDiag = dfInfo.loc[(strSub, strSes), 'DX_Scan']
    strOrigPath = dfInfo.loc[(strSub, strSes), 'T1w_Path']
        
    strCoregPath = os.path.join(strImageDir, 
                                f'sub-{strSub}_ses-{strSes}_space-MNI152NLin2009cAsym_desc-brain_T1w.nii.gz')
    if not os.path.exists(strCoregPath):
        print('No preprocessed image found in', strImageDir)
        lsMetrics += [{}]
        continue
    
    strMaskPath = strCoregPath.replace('T1w', 'mask')
    
    with open(strOrigPath.replace('nii.gz', 'json'), 'r') as f:
        dictHeader = json.load(f)
    
    imgOrig = nibabel.load(strOrigPath)
    imgCoreg = nibabel.load(strCoregPath)
    imgMask = nibabel.load(strMaskPath)
    imgHippoResamp = nilearn.image.resample_to_img(imgHippo, imgCoreg, interpolation='nearest')
    
    arrCoreg = np.array(imgCoreg.dataobj)
    arrMask = np.array(imgMask.dataobj, dtype=np.bool)
    arrHippoResamp = np.array(imgHippoResamp.dataobj, dtype=np.bool)
    
    fBrainMean = arrCoreg[arrMask].mean()
    fHippoMean = arrCoreg[arrHippoResamp].mean()
    fBrainSD = arrCoreg[arrMask].std()
    
    # Get the inner and outer edges of the hippocampus using binary erosion and dilation
    arrBall = morphology.generate_binary_structure(3, 2)
    arrHippoInner = arrHippoResamp ^ morphology.binary_erosion(arrHippoResamp, arrBall)
    arrHippoOuter = morphology.binary_dilation(arrHippoResamp, arrBall) ^ arrHippoResamp
    # Distance transform: for each background voxel, get index of nearest edge voxel
    arrNearestOuterVoxel = morphology.distance_transform_edt(~arrHippoOuter, return_distances=False, 
                                                             return_indices=True)
    # Find index of nearest outer edge voxel for each inner edge voxel
    arrCorrespOuterVoxel = arrNearestOuterVoxel[:, arrHippoInner]
    arrOuterIntensity = arrCoreg[arrCorrespOuterVoxel[0, :], arrCorrespOuterVoxel[1, :], arrCorrespOuterVoxel[2, :]]
    arrInnerIntensity = arrCoreg[arrHippoInner]
    fHippoEdgeContrast = (arrOuterIntensity - arrInnerIntensity).mean()

    dictMetrics = {'Diag': strDiag,
                'T1w_Path': strOrigPath,
                'Site': strSite,
                # 'Manufacturer': dictHeader['Manufacturer'],
                'Model': dictHeader['ManufacturersModelName'],
                'Series_Description': dictHeader['SeriesDescription'],
                'Slice_Thickness': dictHeader['SliceThickness'],
                'TE': dictHeader['EchoTime'],
                'TR': dictHeader['RepetitionTime'],
                # 'TI': dictHeader['InversionTime'],
                'Flip_Angle': dictHeader['FlipAngle'],
                'Voxel_Size_X': imgOrig.affine[0, 0],
                'Voxel_Size_Y': imgOrig.affine[1, 1],
                'Voxel_Size_Z': imgOrig.affine[2, 2],
                'Hippocampus_Mean_Intensity': fHippoMean,
                'Brain_Mean_Intensity': fBrainMean,
                'Brain_SD_Intensity': fBrainSD,
                'Hippocampus_Edge_Contrast': fHippoEdgeContrast
                }
     
    if 'Manufacturer' in dictHeader.keys():
        dictMetrics['Manufacturer'] = dictHeader['Manufacturer']
    if 'InversionTime' in dictHeader.keys():
        dictMetrics['TI'] = dictHeader['InversionTime']
        
    lsMetrics += [dictMetrics]
    
dfMetrics = pd.DataFrame(lsMetrics)
dfMetrics.to_csv('image_info_quality_metrics.csv')
