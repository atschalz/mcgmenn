'''
Extract coronal, axial, and sagittal cross-sections of the hippocampus from skullstripped images.
'''
import warnings
import os
import re
import glob
import shutil
import tempfile
import tqdm
import nilearn.image
import nilearn.plotting
import pandas as pd
import numpy as np
from scipy import ndimage
from PIL import Image
from nipype.interfaces import ants
from armed.misc import expand_data_path

# ignore MatplotlibDeprecationWarning caused by nilearn
warnings.filterwarnings(action='ignore', category=UserWarning)

# Preprocessed image root
strDataPath = expand_data_path('ADNI23_sMRI/DLLabPipeline_sMRI_20220103')
# MNI coordinates for right hippocampus
tupHippoCoordsMNI = (32, -6, -26)
# ANTS installed path
strAntsPath = '/project/bioinformatics/DLLab/softwares/ants2.3.1_build/bin'
# Output dir
strOutDir = expand_data_path('ADNI23_sMRI/right_hippocampus_slices_2pctnorm', make=True)

# Make output subdirs for each slice direction
strSagDir = os.path.join(strOutDir, f'sagittal_MNI{tupHippoCoordsMNI[0]}')
os.makedirs(strSagDir, exist_ok=True)
strCorDir = os.path.join(strOutDir, f'coronal_MNI{tupHippoCoordsMNI[1]}')
os.makedirs(strCorDir, exist_ok=True)
strAxDir = os.path.join(strOutDir, f'axial_MNI{tupHippoCoordsMNI[2]}')
os.makedirs(strAxDir, exist_ok=True)

# Load image info with diagnoses
dfInfo = pd.read_csv('image_list_ad_cn.csv', index_col=0)
dfInfo['RID'] = dfInfo['RID'].apply(lambda x: f'{int(x):04d}')
dfInfo['ScanDate'] = dfInfo['ScanDate'].apply(lambda x: x.replace('-', ''))
dfInfo.index = pd.MultiIndex.from_frame(dfInfo[['RID', 'ScanDate']])

# Conversion from MNI (RAS) to ITK (LPS) coordinate systems
def mni2itk(coords):
    return (str(-coords[0]), str(-coords[1]), str(coords[2]))

def itk2mni(coords):
    return (-float(coords[0]), -float(coords[1]), float(coords[2]))

# Convert coords to ITK and store in a temporary CSV, which is needed to use ANTs
strTempDir = tempfile.mkdtemp()
strCoordsPath = os.path.join(strTempDir, 'coords.csv')
tupHippoCoordsITK = mni2itk(tupHippoCoordsMNI)
with open(strCoordsPath, 'w') as f:
    f.write('x,y,z\n')
    f.write(','.join(tupHippoCoordsITK))
    f.write('\n')

# Get list of image directories
lsImageDirs = glob.glob(os.path.join(strDataPath, 'sub*', 'ses*', 'anat'))
lsImageDirs.sort()

# Get slice, crop empty pixels, resample to 192 x 192, and save to PNG
def save_slice(img, coord, outpath, mode='sag'):    
    # Transform slice coordinate to voxel space and get slice from image
    arrAffine = np.linalg.inv(img.affine)
    if mode == 'sag':
        coord_vox = nilearn.image.coord_transform(coord, 0, 0, arrAffine)[0]
        imgSlice = img.slicer[int(coord_vox):int(coord_vox)+1, ...]
    if mode == 'cor':
        coord_vox = nilearn.image.coord_transform(0, coord, 0, arrAffine)[1]
        imgSlice = img.slicer[:, int(coord_vox):int(coord_vox)+1, :]
    if mode == 'ax':
        coord_vox = nilearn.image.coord_transform(0, 0, coord, arrAffine)[2]
        imgSlice = img.slicer[:, :, int(coord_vox):int(coord_vox)+1]
    arrSlice = np.array(imgSlice.get_fdata()).squeeze()
    
    # Crop, leaving perimeter of 2 empty pixels around the brain
    arrDataIdx = np.array(np.where(arrSlice > 1e-6))
    arrStartIdx = np.maximum(arrDataIdx.min(axis=1) - 2, 0)
    arrEndIdx = np.minimum(arrDataIdx.max(axis=1) + 2, arrSlice.shape)
    
    # Pad shorter axis so that image is square
    arrSize = arrEndIdx - arrStartIdx
    iLongAxis = np.argmax(arrSize)
    iShortAxis = np.argmin(arrSize)
    nPad = arrSize[iLongAxis] - arrSize[iShortAxis]
    arrStartIdx[iShortAxis] = np.maximum(arrStartIdx[iShortAxis] - nPad // 2, 0)
    arrEndIdx[iShortAxis] = arrStartIdx[iShortAxis] + arrSize[iLongAxis]
    arrCrop = arrSlice[(slice(arrStartIdx[0], arrEndIdx[0]), slice(arrStartIdx[1], arrEndIdx[1]))]
    
    # Resample to 192x192
    # arrCrop -= arrCrop.min()
    # arrCrop *= (255 / arrCrop.max())
    arrResamp = ndimage.zoom(arrCrop, 192 / arrSize[iLongAxis], order=1)
    arrResamp = np.rot90(arrResamp)
    # Normalize the 2nd and 98th percentiles to 0-255. Using these percentiles
    # accounts for small perturbations like remaining bone which would skew
    # min-max normalization.
    p2, p98 = np.percentile(arrResamp, [2, 98])
    arrResamp -= p2
    arrResamp *= (255 / (p98 - p2))
    arrResamp = np.clip(arrResamp, 0, 255)
    imgResamp = Image.fromarray(np.uint8(arrResamp), mode='L')
    imgResamp.save(outpath)

for strImageDir in tqdm.tqdm(lsImageDirs):
    regmatch = re.search(r'sub-(\d+).*ses-(\d+)', strImageDir)
    strSub = regmatch[1]
    strSes = regmatch[2]

    # Find MNI coregistration warp, affine, and skullstripped image
    strWarpPath = os.path.join(strImageDir, 
                                f'sub-{strSub}_ses-{strSes}_to-MNI152NLin2009cAsym_desc-nonlinear_xfm.nii.gz')
    if not os.path.exists(strWarpPath):
        print('No warp found for', strImageDir)
        continue
    strAffinePath = os.path.join(strImageDir, 
                                f'sub-{strSub}_ses-{strSes}_to-MNI152NLin2009cAsym_desc-affine_xfm.mat')
    if not os.path.exists(strAffinePath):
        print('No affine found for', strImageDir)
        continue
    strImagePath = os.path.join(strImageDir, 
                                f'sub-{strSub}_ses-{strSes}_desc-brain_T1w.nii.gz')
    if not os.path.exists(strImagePath):
        print('No image found for', strImageDir)
        continue

    # ANTS transformation interface
    strCoordsTransformedPath = os.path.join(strTempDir, 'coords_out.csv')
    transform = ants.ApplyTransformsToPoints(input_file=strCoordsPath,
                                            transforms=[strWarpPath, strAffinePath],
                                            dimension=3,
                                            invert_transform_flags=[False, False],
                                            output_file=strCoordsTransformedPath,
                                            environ={'PATH': '$PATH:' + strAntsPath})
    transform.run()

    with open(strCoordsTransformedPath, 'r') as f:
        lsCoordsTransformedITK = f.read().split('\n')[1].split(',')

    tupCoordsTransformedMNI = itk2mni(lsCoordsTransformedITK)
    img = nilearn.image.load_img(strImagePath)
    
    strDiag = dfInfo.loc[(strSub, strSes), 'DX_Scan']
       
    strSliceSagPath = os.path.join(strSagDir, f'sub-{strSub}_ses-{strSes}_{strDiag}.png') 
    save_slice(img, tupCoordsTransformedMNI[0], strSliceSagPath, mode='sag')
    
    strSliceCorPath = os.path.join(strCorDir, f'sub-{strSub}_ses-{strSes}_{strDiag}.png') 
    save_slice(img, tupCoordsTransformedMNI[1], strSliceCorPath, mode='cor')
    
    strSliceAxPath = os.path.join(strAxDir, f'sub-{strSub}_ses-{strSes}_{strDiag}.png') 
    save_slice(img, tupCoordsTransformedMNI[2], strSliceAxPath, mode='ax')

shutil.rmtree(strTempDir)