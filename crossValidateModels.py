# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:17:58 2015

@author: grigoriykoytiger
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import ARDRegression
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_predict
from datetime import date
import os
import argparse

def ranktransform(X):
    from scipy.stats.mstats import rankdata
    rankX = rankdata(X)
    Xpercentile = rankX/ X.shape[0]
    return Xpercentile
    
if __name__ == "__main__":
    
    ########################### Set script paramaters  ###########################
    parser = argparse.ArgumentParser()
    parser.add_argument("min", type=int, help="index to start cross validating model") 
    parser.add_argument("max", type=int, help="index to end cross validating model")    
    args = parser.parse_args()
     
     #Location for training data
    inputFolder = '../output/standardizedData/2015-07-30/'
    outputFolder = '../output/crossValidation/' + np.str(date.today()) + '/predictions/'
       
    #Gene expression paramaters
    geneExpression = joblib.load(inputFolder + 'cellExpressionPrunedNormalized.pkl')
       
    #Drug paramaters    
    IC50 =  joblib.load(inputFolder + 'IC50.pkl')
    drugNames = np.unique(IC50['drug_name'])
    rankTransformIC50 = True

    #Drug Model Paramaters    
    clf = ARDRegression(normalize=False)

    
    #Training data paramaters
    cutoffSamplePoints = 100 #Minimum number of unique cell-lines required to build drug model
            
    if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
    #############################################################################
    
    if rankTransformIC50:
        IC50['IC50']= ranktransform(IC50['IC50'])
       
    for drug in drugNames[args.min:args.max]:
            y= IC50.ix[IC50['drug_name'] == drug]
            X= geneExpression[geneExpression.index.isin(y['cell_line'])]
            preds = cross_val_predict(clf, X, y=y['IC50'].values.ravel(), cv=4, n_jobs=-1)
            output = pd.DataFrame(data=preds, index = X.index)
            output.to_csv(outputFolder + drug + '.csv') 
    
