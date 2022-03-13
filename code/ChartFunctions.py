import os
import pandas as pd
import numpy as np
from datetime import datetime

from code.RunFunctions import CheckKeys

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from statistics import mean

def GenerateLineChart(df, column, ax, title): 
    ax.set_title(title)
    sns.lineplot(data = df, x = 'date', y = column, ax = ax)

def GenerateBarChart(df, column, ax, title): 
    ax.set_title(title)
    sns.barplot(data = df, x = 'date', y = column, ax = ax)
    
def GenerateHistogram(df, column, ax, title, marker = 0.0):
    ax.set_title(title)
    data = df[column].dropna()
    sns.kdeplot(data,bw=0.25,ax=ax, fill=True)
    ax.axvline(marker, 0, 1)
    color = 'green'
    if mean(data)<marker:
        color = 'red'
    ax.axvline(mean(data), 0, 1, color = color)
    m = max(abs(min(data)), abs(max(data)))
    #ax.axvline(-m, 0, 0)
    #ax.axvline(m, 0, 0)

def GenerateSummaryParCoords(allocator_returns, allocators, prices, datasets, classifiers, predictors):
    combos, combos['prices'], combos['datasets'], combos['classifiers'], combos['traders'], combos['allocators'], combos['std_daily'], combos['mean_daily'] = {},[],[],[],[],[],[],[]
    for a in allocator_returns:
        for p in prices:
            for d in datasets:
                if not CheckKeys(False, datasets[d].included, datasets[d].excluded, p):
                    continue
                for c in classifiers:
                    if not CheckKeys(False, classifiers[c].included, classifiers[c].excluded, p, d):
                        continue
                    for t in predictors:
                        if not CheckKeys(False, predictors[t].included, predictors[t].excluded, p, d, c):
                            continue
                        if CheckKeys(False, allocators[a].included, allocators[a].excluded, p, d, c, t):
                            combos['prices'].append(p)
                            combos['datasets'].append(d + '_' + p)
                            combos['classifiers'].append(c + '_' + d + '_' + p)
                            combos['traders'].append(t + '_' + c + '_' + d + '_' + p)
                            combos['allocators'].append(a)
                            combos['std_daily'].append(np.std(allocator_returns[a]['weighted_pal']))
                            combos['mean_daily'].append(np.mean(allocator_returns[a]['weighted_pal']))

    items = []
    for key in ['prices', 'datasets', 'classifiers', 'traders', 'allocators']:
        ticktext = list(np.unique(np.array((combos[key]))))
        ticktext.sort(reverse=True)
        tickvals = list(range(0,len(ticktext)))
        values = []
        for p in combos[key]:
            values.append(ticktext.index(p))
        items.append({'tickvals':tickvals,'label':key,'ticktext':ticktext, 'values':values})
    for key in ['mean_daily']: # 'std_daily'
        items.append({'label':key,'values':combos[key]})

    fig = go.Figure(data= go.Parcoords(line = dict(colorscale = 'Electric'), dimensions = items))
    fig.show()

def GenerateCorrelationPlot(allocator_returns):
    cor_dict = {}
    for a in allocator_returns:
        cor_dict[a] = allocator_returns[a]['weighted_pal']
    cor_df = pd.DataFrame(cor_dict)
    corrMatrix = cor_df.corr()

    sns.heatmap(corrMatrix, annot=True, vmin=-1, vmax=1)
    plt.show()
    
#classifier results
def GenerateClassifierResultsChart(prices, datasets, price_dataset_results, classifiers, price_dataset_classifier_results, start_date, end_date):
    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    count = 0
    classifier_names, ma_twenty_dps = [], []
    for p in prices:
        for d in datasets:
            if not CheckKeys(False, datasets[d].included, datasets[d].excluded, p):
                continue
                
            d_sind, d_eind = price_dataset_results[p][d].index.get_loc(start_date), price_dataset_results[p][d].index.get_loc(end_date)
            all_labs = price_dataset_results[p][d]['label'].to_numpy()
            labs = all_labs[d_sind:d_eind+1]
            labs[labs<1] = -1
            
            for c in classifiers:
                if not CheckKeys(False, classifiers[c].included, classifiers[c].excluded, p, d):
                    continue
                c_sind, c_eind = price_dataset_classifier_results[p][d][c].index.get_loc(start_date), price_dataset_classifier_results[p][d][c].index.get_loc(end_date)
                all_preds = price_dataset_classifier_results[p][d][c]['prediction'].to_numpy()
                preds = all_preds[c_sind:c_eind+1]
                score = (labs==preds)
                ma_twenty = moving_average(score, n = 20)
                ovmen = np.mean(score)
                med = np.median(ma_twenty)
                men = np.mean(ma_twenty)
                print(c + '_' + d + '_' + p + f': overall mean: {ovmen:.3f}, median ma_twenty: {med:.2f}, mean ma_twenty: {men:.3f}')

                for i in range(len(ma_twenty)):
                    classifier_names.append(c + '_' + d + '_' + p)
                    ma_twenty_dps.append(ma_twenty[i])
                count += 1
                    
    sns.set(style="whitegrid")  
    y_size = count * 1.33
    plt.figure(figsize=(12, y_size))
    ax = sns.violinplot(x=ma_twenty_dps, y=classifier_names, saturation=0.4)
