# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 20:14:36 2015

@author: grigoriykoytiger
"""

import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
outputFolder = '../output/Presentation/'

allPredictions = pd.read_csv('../output/tcga/2015-07-23/all_predictions.csv', index_col =0, na_values = ['[Not Available]', '[Not Applicable]', '[Unknown]'])
allPredictions = allPredictions.ix[~np.isnan(allPredictions['AFATINIB']),:]

gemcitabineResults = allPredictions.ix[allPredictions['pharmaceutical_therapy_drug_name'] == 'GEMCITABINE',:]
gemcitabineResultsLung = gemcitabineResults.ix[(gemcitabineResults['cancer_type'] == 'LUAD') | (gemcitabineResults['cancer_type'] == 'LUSC'),:]

ax = sns.boxplot(x="tumor_status", y='GEMCITABINE', data=gemcitabineResultsLung)    
ax = sns.stripplot(x="tumor_status", y='GEMCITABINE', data=gemcitabineResultsLung, jitter=False, edgecolor="gray", alpha=.7, size=5)
figure = ax.get_figure()
figure.savefig(outputFolder + 'GEMCITABINE' + '_tumor_status.pdf')
ax.clear()

withTumorPrediction = gemcitabineResultsLung.ix[gemcitabineResultsLung['tumor_status'] == 'WITH TUMOR', 'GEMCITABINE']
withoutTumorPrediction = gemcitabineResultsLung.ix[gemcitabineResultsLung['tumor_status'] == 'TUMOR FREE', 'GEMCITABINE']
stats.ranksums(withTumorPrediction, withoutTumorPrediction)

roc_auc_score(np.hstack((np.repeat(0,6), np.repeat(1,12))), np.hstack((withoutTumorPrediction, withTumorPrediction)))
roc_data = pd.DataFrame()
roc_data['fpr'], roc_data['tpr'],roc_data['thresholds'] = roc_curve(np.hstack((np.repeat(0,6), np.repeat(1,12))), np.hstack((withoutTumorPrediction, withTumorPrediction)))
roc_data.plot(x='fpr', y='tpr').get_figure().savefig(outputFolder + 'gemcitabine_ROC_Curve.pdf')



doxorubicinResults = allPredictions.ix[allPredictions['pharmaceutical_therapy_drug_name'] == 'DOXORUBICIN',:]


#Cross validation graphic
input_dir = '2015-07-20/cross_validation/'
cross_validation_statistics = pd.read_csv(input_dir + 'cross_validation_statistics.csv')

figure = sns.distplot(cross_validation_statistics['Spearman rho'], bins=[-.05, 0, .05, .1, .15, .2, .25, .3, .35], kde=False).get_figure()

figure.savefig(outputFolder + 'cross_validation_histogram.pdf')
