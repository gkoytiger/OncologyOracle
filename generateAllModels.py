# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:17:58 2015

@author: grigoriykoytiger
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import ARDRegression
from sklearn.externals import joblib
from datetime import date
from crossValidateModels import ranktransform
import argparse
import os


def train_predictor(drug, outputFolder, clf,expression, IC50):
    expression = expression[expression.index.isin(IC50['cell_line'])]
    clf = clf.fit(expression.values,IC50['IC50'].values.ravel())
    clf.sigma_=[] #wipes the feature correlation matrix to dramaticaly shrink stored model 
    joblib.dump(clf, outputFolder + drug, compress=3)
    
    
if __name__ == "__main__":
    
    ########################### Set script paramaters  ###########################
    parser = argparse.ArgumentParser()
    parser.add_argument("min", type=int, help="index to start training model") 
    parser.add_argument("max", type=int, help="index to end training model")    
    args = parser.parse_args()

         
    #Drug Model Paramaters    
    clf = ARDRegression(normalize=False)
    
    #Location for training data
    inputFolder = '../output/standardizedData/2015-07-30/'
    outputFolder = '../output/models/' + np.str(date.today()) + '/'
    validatedDrugs = joblib.load('../output/crossValidation/2015-08-01/validatedDrugs.pkl')
    
    
    #Training data paramaters
    rankTransformIC50 = True
    geneExpression = joblib.load(inputFolder + 'cellExpressionPrunedNormalized.pkl')
    IC50 =  joblib.load(inputFolder + 'IC50.pkl')
    
    if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
    #############################################################################
    if rankTransformIC50:
        IC50['IC50']= ranktransform(IC50['IC50'])
        
    for drug in validatedDrugs[args.min:args.max]:
        train_predictor(drug, outputFolder, clf, geneExpression, IC50.ix[IC50['drug_name'] == drug])
        
