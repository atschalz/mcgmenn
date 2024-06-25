"""
Get pertinent variables and prediction labels (conversion from MCI w/in X months) from the ADNIMERGE table. 
"""

TABLE = 'ADNIMERGE.csv'
MONTHS = 24 # Look for conversion within this many months

import pandas as pd
import numpy as np

dfData = pd.read_csv(TABLE, index_col='PTID')

# Remove the ">" and "<" values for the CSF measurements, where something was
# outside detection limits. Just assume that the value == the minimum measurable
# value. 
dfData.replace('>1700', 1700, inplace=True)
dfData.replace('>1300', 1300, inplace=True)
dfData.replace('>120', 120, inplace=True)
dfData.replace('<200', 200, inplace=True)
dfData.replace('<80', 80, inplace=True)
dfData.replace('<8', 8, inplace=True)

for strCol in ['ABETA', 'ABETA_bl', 'TAU', 'TAU_bl', 'PTAU', 'PTAU_bl']:
    dfData[strCol] = dfData[strCol].astype(np.float32)


# Attempt to infer missing diagnosis values based on the neighboring visits if
# they have matching diagnoses. E.g. if month 24 is missing but months 18 and 30
# are both 'MCI', assume that month 24 is also 'MCI'
lsDataImputed = []
for strSubject in dfData.index.unique():
    dfDataSubject = dfData.loc[strSubject].copy()
    if len(dfDataSubject.shape) == 1:
        dfDataSubject.loc['DX_impute'] = dfDataSubject.loc['DX']
        lsDataImputed += [dfDataSubject.to_frame().T]
    else:
        dfDataSubject = dfDataSubject.sort_values('Month')
        
        dfIsNan = dfDataSubject['DX'].isna()
        dfPrevVal = dfDataSubject['DX'].ffill()
        dfNextVal = dfDataSubject['DX'].bfill()
        dfPrevNextEqual = (dfPrevVal == dfNextVal)
        dfDataSubject['DX_impute'] = dfDataSubject['DX'].mask(dfIsNan & dfPrevNextEqual, dfPrevVal)
        lsDataImputed += [dfDataSubject]
    
dfDataImputed = pd.concat(lsDataImputed)

# Select observations for subjects who had MCI at baseline
dfMCIData = dfDataImputed.loc[dfDataImputed['DX_bl'].isin(['LMCI', 'EMCI'])]
subsMCI = dfMCIData.index.unique()
print(subsMCI.shape[0], 'subjects with MCI at baseline')

# Check whether these subjects converted to AD within X months
lsConverters = []
lsNonconverters = []
for strSub in subsMCI.values:
    dfSubjectVisits = dfDataImputed.loc[strSub]

    if len(dfSubjectVisits.shape) == 1:
        dfSubjectVisits = dfSubjectVisits.to_frame().T
    
    if any(dfSubjectVisits['DX_impute'].loc[dfSubjectVisits['Month'] <= MONTHS] == 'Dementia'):
        lsConverters += [strSub]
        
    # Check for a visit exactly at the target month
    elif MONTHS in dfSubjectVisits['Month']:
        if dfSubjectVisits['DX_impute'].loc[dfSubjectVisits['Month'] == MONTHS] != 'Dementia':
            lsNonconverters += [strSub]
            
    # Check the next available visit
    else:
        dfNextVisits = dfSubjectVisits.loc[dfSubjectVisits['Month'] > MONTHS]
        if dfNextVisits.shape[0] > 0:
            if dfNextVisits['DX_impute'].iloc[0] != 'Dementia':
                lsNonconverters += [strSub]
            
print(len(lsConverters), 'converters')
print(len(lsNonconverters), 'nonconverters')

# Series of class labels
dfConversion = pd.Series(index=lsConverters + lsNonconverters, dtype=np.float32)
dfConversion.loc[lsConverters] = 1
dfConversion.loc[lsNonconverters] = 0

# Get the baseline features 
dfFeatures = dfData.iloc[:, [x.endswith('_bl') for x in dfData.columns]]
# Add the demographics too
dfDemo = dfData[['VISCODE', 'SITE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4']]
dfFeatures = pd.concat([dfDemo, dfFeatures], axis=1)

# Keep only unique subjects
dfFeatures = dfFeatures.loc[~dfFeatures.index.duplicated(keep='first')]
dfFeatures = dfFeatures.loc[dfConversion.index]
dfFeatures.pop('DX_bl')

# dfConversion.to_csv(f'target_{MONTHS}mo_imputeddx.csv')
# dfFeatures.to_csv(f'baseline_features_{MONTHS}mo_imputeddx.csv')