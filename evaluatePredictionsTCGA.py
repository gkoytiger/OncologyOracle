# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:54:32 2015

@author: grigoriykoytiger
"""

import os
import pandas as pd
import numpy as np
import glob
from datetime import date
from sklearn.externals import joblib
import plotly.plotly as py
from plotly.graph_objs import *
import scipy.stats as stats


def fixSampleIDs(predictions):
    predictions = predictions.reset_index()
    predictions['index'] = predictions['index'].apply(lambda x: x[:12])
    
    #Takes 01 sample instead of 11
    predictions = predictions.groupby(by=['index']).last()
    
    return predictions

def doPredictions(drug, modelFolder, tcgaExpression):
    #Runs predictions, linearly rescales predictions for a drug so they are all between 0,1
    clf = joblib.load(modelFolder + drug)
    predictions = clf.predict(tcgaExpression)
    minPrediction = np.min(predictions)
    maxPrediction = np.max(predictions)
    rescaledPredictions = (predictions -  minPrediction)/(maxPrediction-minPrediction)
    return rescaledPredictions, minPrediction, maxPrediction
    
def ingestDrugData(tcgaDataFolder, cancerType, validatedDrugs):
    #Takes in TCGA clinical drug file and extracts whether a patient went into remission with a course of therapy
    filename = glob.glob(tcgaDataFolder + 'clinical/*' + cancerType + '*/*drug*')
    tcgaDrugData = pd.read_csv(filename[0], sep='\t', header = [0,2], na_values = ['[Not Available]', '[Not Applicable]', '[Unknown]'])
    tcgaDrugData = tcgaDrugData[['bcr_patient_barcode', 'pharmaceutical_therapy_drug_name', 'treatment_best_response', 'pharmaceutical_tx_ongoing_indicator']]    
    
    #Drop multi index, capitalizate drug name
    tcgaDrugData.columns = tcgaDrugData.columns.droplevel(level=[1])    
    tcgaDrugData['pharmaceutical_therapy_drug_name'] = tcgaDrugData['pharmaceutical_therapy_drug_name'].str.capitalize()
    
    #Filter out drugs not modelled
    tcgaDrugData = tcgaDrugData.ix[tcgaDrugData['pharmaceutical_therapy_drug_name'].isin(validatedDrugs),:].sort(columns=['pharmaceutical_therapy_drug_name', 'treatment_best_response'])
    return tcgaDrugData
    
def ingestPatientData(tcgaDataFolder, cancerType, patientIDs):
    filename = glob.glob(tcgaDataFolder + 'clinical/*' + cancerType + '*/*patient*')

    #Reads in patient level data, extracts clinical status factors
    tcgaPatientData = pd.read_csv(filename[0], sep='\t', header = [0,2], na_values = ['[Not Available]', '[Not Applicable]', '[Unknown]'])
    
    #removes multi-index
    tcgaPatientData.columns = tcgaPatientData.columns.droplevel(level=[1])    
    tcgaPatientData = tcgaPatientData[['bcr_patient_barcode', 'tumor_status', 'vital_status', 'death_days_to']]    
    
    #Filter out patients without any drug data
    tcgaPatientData = tcgaPatientData.ix[tcgaPatientData['bcr_patient_barcode'].isin(patientIDs),:]
    return tcgaPatientData


def compareDrug(allDataMerged, drugName, outputFolder, cancerType):
    
    drugData = allDataMerged.ix[allDataMerged['pharmaceutical_therapy_drug_name'].isin([drugName]),:]
    
    #Compare whether we can predict who will have have no tumor
    withoutTumorPrediction = drugData.ix[drugData['tumor_status'].isin(['TUMOR FREE']), drugName].values
    withTumorPrediction =drugData.ix[drugData['tumor_status'].isin(['WITH TUMOR']), drugName].values
    p= stats.ranksums(withTumorPrediction, withoutTumorPrediction)[1]
    
    
    trace0 = Box(
        y=withoutTumorPrediction,
        name='Tumor Free',
        jitter=0.1,
        marker=Marker(
            color='#3D9970'
        ),
        boxpoints='all'
    )

    trace1 = Box(
        y=withTumorPrediction,
        name='With Tumor',
        jitter=0.1,
        marker=Marker(
            color='#FF4136'
        ),
        boxpoints='all'
    )    

    data = Data([trace0, trace1])
    layout = Layout(
        yaxis=YAxis(
            title='Predicted Sensitivity',
            zeroline=False
        ),
        xaxis=XAxis(
            title='Wilcoxon p = ' + np.str(p)[:4],
            zeroline=False
        ),
    )
    
    fig = Figure(data=data, layout=layout)
    plot_url = py.plot(fig, filename= 'TCGA / ' + cancerType + ' / ' + drugName)    

    return

    
    

if __name__ == "__main__":
    
    ########################### Set script paramaters  ###########################
    modelFolder = '../output/models/2015-08-02/'
    tcgaDataFolder = '../data/tcga/'
    tcgaExpression = joblib.load('../output/standardizedData/2015-07-30/tcgaExpressionPrunedNormalized.pkl')
    tcgaSampleType = joblib.load('../output/standardizedData/2015-07-30/tcgaSampleData.pkl')
    validatedDrugs = joblib.load('../output/crossValidation/2015-08-01/validatedDrugs.pkl')
    
    cancerTypes = np.unique(tcgaSampleType)
   
    outputFolder = '../output/tcga/' + np.str(date.today()) + '/'
    if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
    #############################################################################
    
    #Predicts all drug models against all TCGA RNAseq RPKM samples
    allPredictions = pd.DataFrame(index = validatedDrugs, columns=tcgaSampleType.index)
    rescaleFactors = pd.DataFrame(index=validatedDrugs, columns = ['min', 'max'])
    
    for drug in validatedDrugs:
        allPredictions.ix[drug,:], rescaleFactors[drug,'min'], rescaleFactors[drug,'max'] = doPredictions(drug, modelFolder, tcgaExpression)
        
    #Take patient samples, convert to patient IDs for comparison
    allPredictions = allPredictions.T
    allPredictions['cancerType']= tcgaSampleType.values
    allPredictions = fixSampleIDs(allPredictions)
    joblib.dump(allPredictions, outputFolder + 'allPredictions.pkl', compress=3)
    
    allPredictions = allPredictions.groupby(by='cancerType')
    
    allDrugComparisons = pd.DataFrame()
    for cancerType in cancerTypes:
        #Get predictions corresponding to a cancer patient
        
        cancerPredictions = allPredictions.get_group(cancerType)
        
        #Ingest clinical data    
        tcgaDrugData = ingestDrugData(tcgaDataFolder, cancerType, validatedDrugs)
        tcgaPatientData = ingestPatientData(tcgaDataFolder, cancerType, tcgaDrugData['bcr_patient_barcode'])
        
        #merge all data sets
        #outer join treatment, since sometimes patients are treated with multiple drugs, inner join sensitivity predictions
        allDataMerged  = pd.merge(cancerPredictions, pd.merge(tcgaPatientData, tcgaDrugData, how='outer', on='bcr_patient_barcode' ),  how='inner', right_on = 'bcr_patient_barcode', left_index=True)
        allDataMerged.index = allDataMerged.ix[:,'bcr_patient_barcode']

        #run comparison for all drugs used in clinic    
        outputFolderCancer = outputFolder + cancerType + '/'   
        
        if not os.path.exists(outputFolderCancer):
            os.makedirs(outputFolderCancer)    
        
        drugs = np.unique(allDataMerged['pharmaceutical_therapy_drug_name'])
        
        for drugName in drugs:
            compareDrug(allDataMerged, drugName, outputFolderCancer, cancerType)
        
        joblib.dump(allDataMerged, outputFolderCancer + 'allDataMerged.pkl')
        allDrugComparisons = allDrugComparisons.append(allDataMerged)
    
    #Do analaysis over whole dataset
    outputFolderCancer = outputFolder + 'PANCAN/'
    drugs =     np.unique(allDrugComparisons['pharmaceutical_therapy_drug_name'])
    
    if not os.path.exists(outputFolderCancer):
            os.makedirs(outputFolderCancer)  
    
    joblib.dump(allDrugComparisons, outputFolderCancer + 'allDataMerged.pkl', compress=3)
        
    for drugName in drugs:
         compareDrug(allDrugComparisons, drugName, outputFolderCancer, 'PANCAN')
    
    
    #Do analysis for Gemcitabine in lung cancer samples
    
    gemcitabineResults = allDrugComparisons.ix[allDrugComparisons['pharmaceutical_therapy_drug_name'] == 'Gemcitabine',:]
    gemcitabineResultsLung = gemcitabineResults.ix[(gemcitabineResults['cancerType'] == 'LUAD') | (gemcitabineResults['cancerType'] == 'LUSC'),:]
    compareDrug(gemcitabineResultsLung, 'Gemcitabine', outputFolderCancer, 'PANLUNG')
    
    #Generate random subset for web display
    web_tcga_ids = ['TCGA-AO-A0J3', 'TCGA-AO-A03L', 'TCGA-CJ-5677', 'TCGA-BC-A10Q', 'TCGA-38-4632', 'TCGA-60-2707', 'TCGA-AP-A052']
    outputFolder = outputFolder + 'web/'
    
    if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)  
    
    for sampleID in web_tcga_ids:    
        randomPrediction = allDrugComparisons.ix[allDrugComparisons['bcr_patient_barcode'] == sampleID,validatedDrugs[0]:validatedDrugs[-1]]
        model_results = []
        
        for drug_name in randomPrediction.columns.values:
            model_results.append(dict(name=drug_name, score=np.around(randomPrediction[drug_name].values, decimals=2).tolist().pop()))
        sort_results = sorted(model_results, key=lambda k: k['score']) 
        
        joblib.dump(sort_results, outputFolder + sampleID, compress=3)