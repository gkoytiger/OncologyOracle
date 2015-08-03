# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:23:06 2015

@author: grigoriykoytiger
"""
#Package imports
import pandas as pd
import numpy as np
import mysql.connector
from datetime import date
from sklearn.preprocessing import Normalizer
import os
import glob
from sklearn.externals import joblib

def standardizeCellLineNames(cellLineNames):
    #Removes non-letter characters that often vary between data sources
    cellLineNames = cellLineNames.replace(to_replace=['\W'], value='', regex=True).str.upper()
    return cellLineNames.replace(to_replace=['CELLLINE'], value='', regex=True)

def ingestGenentechRPKM(cellExpressionFolder, geneMap):
    #Code to import Genentech published RNA-Seq data
    cellExpression = pd.read_csv(cellExpressionFolder + '140625_Klijn_RPKM_coding.txt', sep='\t')
    cellInformation = pd.read_csv(cellExpressionFolder + 'E-MTAB-2706.sdrf.txt', sep='\t')
    
    cellExpression = pd.merge(geneMap, cellExpression, how='inner', on='geneID')
    cellExpression.index = cellExpression['geneName'].values
    cellExpression.drop(['geneID', 'geneName'], axis=1, inplace=True)
    
    #Sort sample numbers, flip to machine learning standard direction
    cellExpression = cellExpression.reindex_axis(sorted(cellExpression.columns.values, key=lambda x: float(x[7:])), axis=1).T
    cellExpression.index = standardizeCellLineNames(cellInformation['Characteristics[cell line]'])
    cellExpression.sort_index(axis=0, inplace=True)
    
    return cellExpression

def ingestTcgaRPKM(tcgaExpressionFolder, cancerType):
    #Takes Raw TCGA RPKM gene expression data, returns in matrix suitable for model evaluation
    #Use glob to get rid of listing .DS_Store    
    filename = glob.glob(tcgaExpressionFolder + cancerType + '/*')[0]
    
    #Read in TCGA file, use tab or | as delimiters
    data = pd.read_csv(filename, sep='\||\t', header = [0,1], engine='python')
    data = data.xs('RPKM', axis=1, level=1)

    return data.T

def standardizeExpression(expressionData, Norm, log10Normalize):
   
    if log10Normalize:
           expressionData = np.log10(expressionData + 1)
    
    standardizedValues = Norm.fit_transform(expressionData)
    normalizedDataFrame = pd.DataFrame(data=standardizedValues, index=expressionData.index.values, columns=expressionData.columns.values)
    return normalizedDataFrame, Norm
    
def ingestAllTcgaExpressionData(tcgaExpressionFolder):
    cancerTypes = os.listdir(tcgaExpressionFolder)[1:] #removes .DS_Store file name
    
    tcgaExpression = pd.DataFrame()
    tcgaSampleData = pd.DataFrame()

    for cancerType in cancerTypes:
        cancerData = ingestTcgaRPKM(tcgaExpressionFolder, cancerType)
        cancerData['cancerType'] = cancerType
        tcgaExpression = tcgaExpression.append(cancerData)

    
    return tcgaExpression['cancerType'], tcgaExpression.drop(['cancerType', '?'], axis=1)

def getChembl():

    cnx = mysql.connector.connect(user='root', password='')
    chemblData = pd.read_sql_query("SELECT assay_cell_type AS cell_line, pref_name AS drug_name, standard_value AS IC50 from data.human_cell_ic50", cnx)
    cnx.close()
    chemblData = standardizeDrugNames(chemblData)
    chemblData['cell_line'] = standardizeCellLineNames(chemblData['cell_line'])

    return chemblData
   
def getCustom():
    
    cnx = mysql.connector.connect(user='root', password='')
    customData = pd.read_sql_query('SELECT cell_line_id, drug_id, value*1000 AS IC50 FROM custom_data.cell_line_drug WHERE metric_id = 5;', cnx)
    cell_line_reference = pd.read_sql_query('SELECT cell_line_id, cell_line_name AS cell_line FROM custom_data.cell_line_reference;', cnx)
    drug_reference = pd.read_sql_query('SELECT drug_id, drug_name FROM custom_data.drug_reference;', cnx)
    cnx.close()
    
    customData = pd.merge(customData, drug_reference, how='left', on='drug_id').drop(['drug_id'], axis=1)
    customData = pd.merge(customData, cell_line_reference, how='left', on='cell_line_id').drop(['cell_line_id'], axis=1)
    return customData
    
    
def standardizeDrugNames(IC50):
    IC50['drug_name'] = IC50['drug_name'].str.split(' ', expand=True)[0].str.capitalize()
    return IC50
    
def pruneCellLines(IC50, cutoffSamplePoints):
    drug_count = IC50.groupby(by='drug_name').count()
    filtered_drugs = drug_count[drug_count['cell_line']>cutoffSamplePoints].index.values
    return IC50[IC50['drug_name'].isin(filtered_drugs)]

def pruneGenes(geneExpression, numberOfGenesToKeep):
    gene_std = np.std(geneExpression)
    gene_std.sort(axis=1)
    gene_filter = gene_std[-numberOfGenesToKeep:]

    return geneExpression.ix[:,geneExpression.columns.isin(gene_filter.index)]

if __name__ == "__main__":
    
    ########################### Set script paramaters  ###########################
    #Gene expression paramaters
    log10Normalize = True
    standardizeByTCGA = True #Normalize expression data in a unified way for TCGA and cell lines
    L2Normalizer = Normalizer(norm='l2', copy=True) #Method to normalize gene expression values
    cellExpressionFolder = '../data/expression/genentech/'
    tcgaExpressionFolder = '../data/tcga/expression/'
    geneMap = pd.read_csv('../data/tcga/tcgaGeneMap.txt', delimiter='\t')
    
    
    #IC50 paramaters
    useChembl = True #Use locally hosted chembly sql database (must download and import)
    useCustom = True #Use locally hosted drug IC50 values in sql (collected independently)
    log10NormalizeIC50 = True #Transform IC50 into log IC50
    
    #Gene Pruning parameters
    pruneInvariantGenes = True;
    numberOfGenesToKeep = 8000;
    
    #Location for training data
    outputFolder = '../output/standardizedData/' + np.str(date.today()) + '/'
    if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
    #############################################################################


    '''
    Gene Expression Processing
    '''
    #Read in TCGA data
    cellExpression = ingestGenentechRPKM(cellExpressionFolder, geneMap)
    tcgaSampleData, tcgaExpression = ingestAllTcgaExpressionData(tcgaExpressionFolder)
    
    mergedExpression = pd.merge(cellExpression.T, tcgaExpression.T, how='inner', left_index=True, right_index=True).groupby(level=0).first().T

    #Reduce gene feature set, normalize genes
    if pruneInvariantGenes:
        geneExpressionPruned = pruneGenes(mergedExpression, numberOfGenesToKeep)
        geneExpressionNormalized, L2Normalizer = standardizeExpression(geneExpressionPruned, L2Normalizer, log10Normalize)
       
    #Save datasets for downstream analysis
    joblib.dump(tcgaSampleData, outputFolder + 'tcgaSampleData.pkl', compress=3)
    joblib.dump(L2Normalizer, outputFolder +  'L2Normalizer.pkl')
    cell_ix = cellExpression.shape[0]
    
    joblib.dump(mergedExpression.ix[:cell_ix,:],  outputFolder + 'cellExpression.pkl', compress=3)
    joblib.dump(mergedExpression.ix[cell_ix:,:],  outputFolder + 'tcgaExpression.pkl', compress=3)
    
    joblib.dump(geneExpressionPruned.ix[:cell_ix,:],  outputFolder + 'cellExpressionPruned.pkl', compress=3)
    joblib.dump(geneExpressionPruned.ix[cell_ix:,:],  outputFolder + 'tcgaExpressionPruned.pkl', compress=3)
    
    joblib.dump(geneExpressionNormalized.ix[:cell_ix,:],  outputFolder + 'cellExpressionPrunedNormalized.pkl', compress=3)
    joblib.dump(geneExpressionNormalized.ix[cell_ix:,:],  outputFolder + 'tcgaExpressionPrunedNormalized.pkl', compress=3)
    
    
    '''
    Drug IC50 Processing
    '''
    
    chemblIC50 = getChembl()
    customIC50 = getCustom()
    
    IC50 = chemblIC50.append(customIC50)
    
    #Fixes mitomycin 
    IC50.ix[IC50['drug_name'] == 'Mitomycin', 'drug_name'] = 'Mitomycin C'

    #Merge data replicates by median
    IC50 = IC50.groupby(['drug_name', 'cell_line'], as_index=False).median()
    IC50 = IC50[IC50['cell_line'].isin(cellExpression.index)]
    
    IC50 = pruneCellLines(IC50, cutoffSamplePoints)
    

    if log10NormalizeIC50:
        IC50['IC50'] = np.log10(IC50['IC50'])
    
    
    joblib.dump(IC50, outputFolder + 'IC50.pkl', compress=3)