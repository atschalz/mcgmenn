'''
Logistic regression and logistic mixed effects models built on statsmodels.
'''
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

class LogisticGLM:
    def __init__(self, formula: str) -> None:
        """Logistic GLM

        Args:
            formula (str): statsmodels-style formula
        """                    
        self.strFormula = formula
        self.model = None
        
    def fit(self, dataframe: pd.DataFrame):
        """Fit model

        Args:
            dataframe (pd.DataFrame): contains columns for each feature and label
        """        
        self.model = sm.GLM.from_formula(self.strFormula, dataframe, family=sm.families.Binomial())
        self.result = self.model.fit()
        
    def predict(self, dataframe: pd.DataFrame):
        """Predict on data

        Args:
            dataframe (pd.DataFrame): contains same columns as training data

        Raises:
            UserWarning: .fit() has not been called

        Returns:
            pd.DataFrame: predictions
        """        
        if self.model is None:
            raise UserWarning('Model has not been fit yet.')
        
        return self.result.predict(dataframe)

class MixedLogisticGLM:
    def __init__(self, formula: str, re_dict: dict, cluster_name: str):   
        """Mixed effects logistic GLM

        Args:
            formula (str): statsmodels-style formula
            re_dict (dict): statsmodels-style variance component dictionary, e.g. 
                {'Site_slope': '0 + C(Site):VariableName'}
                or
                {'Site_intercept': '0 + C(Site)'}
            cluster_name (str): name of clustering variable
        """                 
        self.strFormula = formula
        self.dictRandomEffects = re_dict        
        self.strClusterName = cluster_name
        self.model = None

    def fit(self, dataframe: pd.DataFrame):    
        """Fit model

        Args:
            dataframe (pd.DataFrame): contains columns for each feature and label
        """           
        self.model = BinomialBayesMixedGLM.from_formula(self.strFormula, self.dictRandomEffects, dataframe)
        self.result = self.model.fit_vb()

    def predict(self, dataframe: pd.DataFrame):
        """Predict on data. Random effects are applied if the cluster has been seen during training.

        Args:
            dataframe (pd.DataFrame): contains same columns as training data

        Raises:
            UserWarning: .fit() has not been called

        Returns:
            pd.DataFrame: predictions
        """        
        if self.model is None:
            raise UserWarning('Model has not been fit yet.')

        # Construct input array
        lsIndep = self.model.fep_names
        arrInputs = np.ones((dataframe.shape[0], len(lsIndep)))
        
        for iVar, strVar in enumerate(lsIndep):
            if strVar != 'Intercept':
                arrInputs[:, iVar] = dataframe[strVar]
                
        # Fixed effect-based predictions (before logit transformation)
        arrPredLinear = self.result.predict(arrInputs, linear=True)
        
        # Get random effects coefficients
        dfRE = self.result.random_effects()
        
        arrRandomEffects = np.zeros((dataframe.shape[0]))
        for i, (_, row) in enumerate(dataframe.iterrows()):
            strCluster = row[self.strClusterName]
            # Find RE's matching this cluster
            strClusterVar = f'C({self.strClusterName})[{strCluster}]'
            dfREFilt = dfRE.filter(like=strClusterVar, axis=0)
            
            if dfREFilt.shape[0] > 0:
                for strRE in dfREFilt.index:
                    lsVars = strRE.split(':')
                    if len(lsVars) == 1:
                        # Add random intercept
                        arrRandomEffects[i] += dfREFilt['Mean'].loc[strRE]
                    else:
                        # Multiply vars with random slope
                        values = [row[x] for x in lsVars[1:]]
                        values += [dfREFilt['Mean'].loc[strRE]]
                        arrRandomEffects[i] += np.product(values)
        
        arrPredMixedLinear = arrPredLinear+ arrRandomEffects

        # Apply logistic link function
        arrMixedPredLogit = self.model.family.link.inverse(arrPredMixedLinear)
        dfPredictionsME = pd.Series(arrMixedPredLogit, index=dataframe.index)
        
        return dfPredictionsME