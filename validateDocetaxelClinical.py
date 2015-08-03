# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 08:45:09 2015

@author: grigoriykoytiger
"""

from generateTrainData import standardizeCellLineNames, standardizeExpression
import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import Normalizer
import os
from sklearn.externals import joblib
from scipy.stats import linregress
from sklearn.externals.joblib import *
from sklearn.linear_model import ARDRegression
from sklearn.metrics import roc_auc_score, roc_curve

def getDrugIC50(drug_name, inputFolder):
    IC50 = joblib.load(inputFolder + 'IC50.pkl')
    IC50 = IC50[IC50['drug_name'] == drug_name].drop('drug_name', axis=1)
    IC50.index=IC50['cell_line']
    return IC50  

def fixNCI60Metadata(geneArrayExpression, docetaxelArrayFolder):
    nci60_metadata = pd.read_csv(docetaxelArrayFolder + 'u95a.fnames.txt', sep='\t')
    nci60_metadata['cell_line'] = nci60_metadata['Validated Names (260302)'].replace(to_replace=['-', '_', ' ', '\.', '..:' ], value='', regex=True).str.upper()
    
    #Maps cell line names
    sampleToLine = nci60_metadata.set_index('RDID')['cell_line'].to_dict()
    return geneArrayExpression.rename_axis(sampleToLine, axis=0)

def getOverlapExpression(geneArrayExpression, rnaSeqExpression):
    array_overlap = geneArrayExpression[geneArrayExpression.index.isin(rnaSeqExpression.index)].sort_index(axis=0)
    array_overlap = array_overlap.ix[:, array_overlap.columns.isin(rnaSeqExpression.columns)]

    rnaseq_overlap = rnaSeqExpression[rnaSeqExpression.index.isin(geneArrayExpression.index)]
    rnaseq_overlap = rnaseq_overlap.ix[:, rnaseq_overlap.columns.isin(geneArrayExpression.columns)]
    
    
    return array_overlap, rnaseq_overlap, array_overlap.columns.values
    
def regressArrayRnaseq(geneArrayExpression, rnaSeqExpression):
    regression_stats = Parallel(n_jobs=-1)(delayed(linregress)(geneArrayExpression.ix[:,x].values, rnaSeqExpression.ix[:,x].values) for x in range(rnaSeqExpression.shape[1]) )
    regression_stats = np.vstack(regression_stats[:])
    return regression_stats

    
if __name__ == "__main__":
    
    ########################### Set script paramaters  ###########################
    #Gene expression paramaters
    log10Normalize = True
    standardizeByTCGA = True #Normalize expression data in a unified way for TCGA and cell lines
    L2Normalizer = Normalizer(norm='l2', copy=True) #Method to normalize gene expression values
    
    
    #Gene Pruning parameters
    pruneUncorrelatedGenes = True; #Eliminates genes that are uncorrelated between array and RNASeq
    pruneCutoff = 0.001; #p value cutoff for pruning
    clinicalSplitPoint =  60 #for the array data, the first 60 entries correspond to NCI60 dataset
   
    clf = ARDRegression(normalize=False)

    #Location for training data
    inputFolder = '../output/standardizedData/2015-07-30/'
    docetaxelArrayFolder = '../data/docetaxel_validation/'
    outputFolder = '../output/DocetaxelClinical/' + np.str(date.today()) + '/'

    if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
    #############################################################################
    
    cellExpression = joblib.load(inputFolder + 'cellExpression.pkl')
    tcgaExpression = joblib.load(inputFolder + 'tcgaExpression.pkl')
    mergedExpression = cellExpression.append(tcgaExpression)
            
    #Retrieves Combat homogonized data for NCI60 cell line and Docetaxel Clincal U95 Array Data
    geneArrayExpression = pd.read_csv(docetaxelArrayFolder + 'DocetaxelHomogonizedExpression.csv', index_col=0).drop('selection',axis=1).sort_index(axis=0)
    geneArrayExpression = fixNCI60Metadata(geneArrayExpression.T, docetaxelArrayFolder)
    
    #Rescales gene array data, finds overlap
    geneArrayExpressionOverlap, rnaSeqExpressionOverlap, overlapGenes = getOverlapExpression(geneArrayExpression, mergedExpression)
    regression_stats = regressArrayRnaseq(geneArrayExpressionOverlap, rnaSeqExpressionOverlap)

    arrayExpressionClinical = geneArrayExpression.ix[clinicalSplitPoint:, geneArrayExpression.columns.isin(overlapGenes)]
    rescaledExpressionClinical = arrayExpressionClinical * regression_stats[:,0] + regression_stats[:,1]
    
    rnaSeqExpression = mergedExpression.ix[:, overlapGenes]
    
    #Where linear regression gives impossible gene expression results (<0) set to 0
    rescaledExpressionClinical[rescaledExpressionClinical<0]=0

    #Use only genes that are statistically correlated
    if pruneUncorrelatedGenes:
        correlatedGenes = regression_stats[:,3] < pruneCutoff
        rescaledExpressionClinical = rescaledExpressionClinical.ix[:,correlatedGenes]
        rnaSeqExpression = rnaSeqExpression.ix[:, correlatedGenes]  
        
    #Train normalizer on RNA seq, apply to rescaled gene expression
    if standardizeByTCGA:    
        rnaSeqExpressionNormalized, L2Normalizer = standardizeExpression(rnaSeqExpression, L2Normalizer, log10Normalize)
        rescaledExpressionClinical = L2Normalizer.transform(np.log10(rescaledExpressionClinical+1))
#    else:
#        prunedRnaSeqExpressionNormalized, L2Normalizer = standardizeExpression(prunedRnaSeqExpression.ix[cellExpression.shape[0],;], L2Normalizer, log10Normalize)
#        prunedArrayExpressionNormalized = L2Normalizer.transform(np.log10(prunedRescaledExpressionClinical+1))

    #Load Docetaxel IC50 Data
    docetaxelData = getDrugIC50('Docetaxel', inputFolder)
    
    #Assemble training data with both IC50 and expression data    
    docetaxelData = pd.merge(docetaxelData, rnaSeqExpressionNormalized, how='inner', left_index=True, right_index=True).drop('cell_line', axis=1)
        
    #Train Docetaxel model    
    clf.fit(docetaxelData.drop(['IC50'], axis=1), docetaxelData['IC50'])    
    
    #Validate on Clinical Data
    resistance_predictions = clf.predict(rescaledExpressionClinical)
    
    #Calculates ROC, first 11 samples correspond to sensitive patients, last 13 are resistant            
    roc_auc_score(np.hstack((np.repeat(0,11), np.repeat(1,13))), resistance_predictions)

    roc_data = pd.DataFrame()
    roc_data['fpr'], roc_data['tpr'],roc_data['thresholds'] = roc_curve(np.hstack((np.repeat(0,11), np.repeat(1,13))), resistance_predictions)


    #Plot Results
    from bokeh.charts import show, output_file
    from bokeh.plotting import figure

    output_file(outputFolder + 'Docetaxel_ROC_Curve_rankIC50.html')
        


    p1 = figure()
    p1.line(
        roc_data['fpr'],                                       # x coordinates
        roc_data['tpr'],                                  # y coordinates
        color='firebrick',                                    # set a color for the line
        legend = 'True Positive Rate',
        line_width = 3                                      
        )
    p1.legend.orientation = "top_left"
    show(p1)