# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 08:47:08 2015

@author: grigoriykoytiger
"""
import os
import pandas as pd
import numpy as np
from seaborn import jointplot
from scipy.stats import pearsonr, spearmanr
from sklearn.externals import joblib
from crossValidateModels import ranktransform

def cross_validation_statistics(drug_name, input_dir, y):
    y_predict = pd.read_csv(input_dir + 'predictions/' + drug_name + '.csv')
    return y_predict.shape[0], np.std(y['IC50']), pearsonr(y_predict['0'].values, y['IC50'].values), spearmanr(y_predict['0'].values, y['IC50'].values)


if __name__ == "__main__":
    
    ########################### Set script paramaters  ###########################
    #Location of 
    inputFolderCrossVal = '../output/crossValidation/2015-08-01/'
    inputFolderTrain = '../output/standardizedData/2015-07-30/'
        
    #Drug paramaters    
    IC50 =  joblib.load(inputFolderTrain + 'IC50.pkl')
    drugNames = np.unique(IC50['drug_name'])
    rankTransformIC50 = True

    #############################################################################

    if rankTransformIC50:
        IC50['IC50']= ranktransform(IC50['IC50'])
        
        
    pearson_rho=[]
    pearson_p = []
    spearman_rho = []
    spearman_p = []
    train_size = []
    train_std = []
    
    
    for drug_name in drugNames:
        train_size_, train_std_, (pearson_rho_, pearson_p_), (spearman_rho_, spearman_p_) = cross_validation_statistics(drug_name, inputFolderCrossVal, IC50.ix[IC50['drug_name'] == drug_name])
        train_size.append(train_size_)    
        pearson_rho.append(pearson_rho_)
        pearson_p.append(pearson_p_)
        spearman_rho.append(spearman_rho_)
        spearman_p.append(spearman_p_)
        train_std.append(train_std_)
    
    cross_validation_statistics = pd.DataFrame(index=drugNames)
    cross_validation_statistics['Train Size']= train_size
    cross_validation_statistics['Train Standard Deviation']= train_std
    cross_validation_statistics['Pearson rho']= pearson_rho
    cross_validation_statistics['Pearson p']= pearson_p
    cross_validation_statistics['Spearman rho']= spearman_rho
    cross_validation_statistics['Spearman p']= spearman_p
    cross_validation_statistics.to_csv(inputFolderCrossVal + 'cross_validation_statistics.csv')
    
    jointplot(np.array(pearson_rho), np.array(train_size)).savefig(inputFolderCrossVal + 'correlation_rho_train_size.pdf')
    jointplot(np.array(pearson_rho), np.array(train_std)).savefig(inputFolderCrossVal + 'correlation_rho_train_std.pdf')
    
    validateDrugs = cross_validation_statistics.index[ (cross_validation_statistics['Spearman rho'] > 0) & (cross_validation_statistics['Spearman p'] < 0.05)].values
    joblib.dump(validateDrugs, inputFolderCrossVal + 'validatedDrugs.pkl', compress=3)