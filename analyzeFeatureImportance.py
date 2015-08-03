# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:11:09 2015

@author: grigoriykoytiger
"""

import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from bokeh.io import output_file, show, vplot
from bokeh.charts import Bar, output_file


def makeFeaturePlots(modelFolder, outputFolder, drugName):
    clf = joblib.load(modelFolder + drugName)
    output_file(outputFolder +  drugName + ".html", title = drugName + ' Features')
    featureDataFrame = joblib.load('../output/emptyDataFrame.pkl')
    featureDataFrame['Feature Importance'] = clf.coef_
    featureDataFrame.sort(columns='Feature Importance', inplace=True)
    
    resistanceGenes= featureDataFrame.index.values[:25]
    resistanceValues= featureDataFrame['Feature Importance'].values[:25]
    
    featureDataFrame.sort(columns='Feature Importance', inplace=True, ascending= False)

    sensitivityGenes = featureDataFrame.index.values[:25]
    sensitivityValues = featureDataFrame['Feature Importance'].values[:25]
    
    
    s1 = Bar(resistanceValues, cat=resistanceGenes.tolist(), title="Top 25 Resistance Genes for " + drugName, xlabel='Genes', ylabel = 'Coefficient', width=800, height=400, tools = False)
    s2 = Bar(sensitivityValues, cat=sensitivityGenes.tolist(), title="Top 25 Senitivity Genes for " + drugName, xlabel='Genes', ylabel = 'Coefficient', width=800, height=400, tools = False, palette = ['#82CC9B'])
    p = vplot(s1, s2)
    show(p)
    
    
########################### Set script paramaters  ###########################

modelFolder = '../output/models/2015-08-02/'
outputFolder = '../output/feature_importance/2015-08-03/'
validatedDrugs = joblib.load('../output/crossValidation/2015-08-01/validatedDrugs.pkl')

#############################################################################


if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)  

for drugName in validatedDrugs:
   makeFeaturePlots(modelFolder, outputFolder, drugName)