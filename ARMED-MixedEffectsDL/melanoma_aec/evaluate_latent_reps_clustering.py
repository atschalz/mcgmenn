'''
Create scatterplots of inter-cluster, intra-PDX latent distance vs.
intra-cluster, inter-PDX latent distance. 
'''

import pandas as pd
import numpy as np
import tqdm
import multiprocessing as mp

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist
from armed.misc import expand_results_path, expand_data_path

import matplotlib.pyplot as plt
import seaborn as sns

strLatentsPath = expand_results_path('melanoma_aec_tiedweights/base_model_v3/epoch010_latents.pkl')
strImageListPath = expand_data_path('melanoma/4pdx6dates/data_train.csv')
strScatterSavePath = expand_results_path('melanoma_aec_tiedweights/base_model_v3/epoch010_latents_distances.svg')

dfLatents = pd.read_pickle(strLatentsPath)
dfImages = pd.read_csv(strImageListPath, index_col=0)

dfLatents -= dfLatents.mean(axis=0)
dfLatents /= dfLatents.std(axis=0)

db = davies_bouldin_score(dfLatents, dfImages['date'])
ch = calinski_harabasz_score(dfLatents, dfImages['date'])

print(f'Clustering scores:'
      f'\n\tDavies-Bouldin (higher is better): {db}'
      f'\n\tCalinski-Harabasz (lower is better): {ch}'
)

with open(strLatentsPath.replace('.pkl', '_clustering.txt'), 'w') as f:
    f.write(f'Davies-Bouldin: {db}'
            f'\nCalinski-Harabasz: {ch}')

dfLatents['Date'] = dfImages['date'].astype(str).values
dfLatents['Cell'] = dfImages['cell'].values
dfLatents['PDX'] = dfImages['celltype'].values

# Compute pairwise Euclidean distances: intra-PDX inter-day and intra-day inter-PDX
print('Computing inter-PDX and inter-date Euclidean distances')
dfDistances = pd.DataFrame(columns=['PDX', 'Other PDX', 'Date0', 'Date1', 'Intra-PDX', 'Inter-PDX'])
arrCellTypes = dfLatents['PDX'].unique()
arrCellTypes.sort()
arrDates = dfLatents['Date'].unique()
arrDates.sort()
i = 0

def mean_str(col):
    # Calculates the mean if values are numeric or the string if values are strings
    if pd.api.types.is_numeric_dtype(col):
        return col.mean()
    else:
        return col.unique() if col.nunique() == 1 else np.nan

def distance(args):
    from_pdx = args['from_pdx']
    from_date = args['from_date']
    to_date = args['to_date']
    # Compute pairwise distances between 
    # 1. Cells from the same day but two different PDXs
    # 2. Cells from the same PDX but two different days
    # Return the mean of the cell pairwise distances for each comparison
    dfLatentPDX = dfLatents.loc[dfLatents['PDX'] == from_pdx]
    # Compute mean latent vector (across frames) for each cell
    dfLatentPDX = dfLatentPDX.groupby('Cell').agg(mean_str)
    
    dfDate0 = dfLatentPDX.loc[dfLatentPDX['Date'] == from_date]
    dfDate0 = dfDate0[range(56)]
    
    dfDate1 = dfLatentPDX.loc[dfLatentPDX['Date'] == to_date]
    dfDate1 = dfDate1[range(56)]
    arrDistIntraPDX = cdist(dfDate0, dfDate1, metric='euclidean')
    fDistIntraPDX = np.mean(arrDistIntraPDX[np.triu_indices_from(arrDistIntraPDX, k=1)])
    
    # Loop through all other PDX's acquired on the same date
    arrToPDX = dfLatents.loc[dfLatents['Date'] == from_date, 'PDX'].unique()
    arrToPDX = arrToPDX[arrToPDX != from_pdx]

    lsInterPDX = []
    for to_pdx in arrToPDX:
        # Compute pairwise distances between each cell of this PDX and each cell other PDXs on the same date
        dfOtherPDX = dfLatents.loc[(dfLatents['PDX'] == to_pdx) & (dfLatents['Date'] == from_date)]
        # Compute mean latent vector across frames for each cell
        dfOtherPDX = dfOtherPDX.groupby('Cell').agg(mean_str)
        dfOtherPDX = dfOtherPDX[range(56)]
        arrDistInterPDX = cdist(dfDate0, dfOtherPDX, metric='euclidean')

        fDistInterPDX = np.mean(arrDistInterPDX[np.triu_indices_from(arrDistInterPDX, k=1)])
        lsInterPDX += [{'PDX': from_pdx, 
                        'Other PDX': to_pdx, 
                        'Date0': from_date, 
                        'Date1': to_date, 
                        'Intra-PDX': fDistIntraPDX, 
                        'Inter-PDX': fDistInterPDX}]
        
    return pd.DataFrame(lsInterPDX)
    
# Build the list of comparisons to be made
lsArgs = []       
for from_pdx in arrCellTypes:
    arrPDXDates = dfLatents.loc[dfLatents['PDX'] == from_pdx, 'Date'].unique()
    for from_date in arrPDXDates:
        arrToDates = arrPDXDates[arrPDXDates != from_date]
        for to_date in arrToDates:
            lsArgs += [{'from_pdx': from_pdx, 'from_date': from_date, 'to_date': to_date}]
        
# Run in parallel
with mp.Pool(8) as pool:
    lsDistances = list(tqdm.tqdm(pool.imap(distance, lsArgs), total=len(lsArgs)))
    
dfDistances = pd.concat(lsDistances)
dfDistancesNorm = dfDistances.copy()
dfDistancesNorm[['Intra-PDX', 'Inter-PDX']] -= dfDistancesNorm[['Intra-PDX', 'Inter-PDX']].values.min()
dfDistancesNorm[['Intra-PDX', 'Inter-PDX']] /= dfDistancesNorm[['Intra-PDX', 'Inter-PDX']].values.max()
fig, ax = plt.subplots(figsize=(7, 7))
sns.scatterplot(data=dfDistancesNorm, x='Intra-PDX', y='Inter-PDX', hue='PDX', style='PDX', 
                hue_order=arrCellTypes,
                style_order=arrCellTypes,
                ax=ax)
ax.set_xlabel('Mean distance between cells from same PDX, different days')
ax.set_ylabel('Mean distance to cells from other PDXs, same day')
minval = dfDistancesNorm[['Intra-PDX', 'Inter-PDX']].values.min()
ax.plot([minval, 1], [minval, 1], 'k--')
fig.savefig(strScatterSavePath)
fig.savefig(strScatterSavePath.replace('svg', 'png'))

print('{:.03f}% of comparisons below diagonal'.format(
    100 * (dfDistancesNorm['Intra-PDX'] > dfDistancesNorm['Inter-PDX']).mean()
    ))
