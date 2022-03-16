import os
import pandas as pd
import numpy as np
from datetime import datetime

from code.classifiers.abstract.Classifier import Classifier

def CheckDefinitions(prices, datasets, classifiers, traders, allocators):
    for c in classifiers:
        print('checking classifier: ' + c)
        if not Classifier().CheckClass(type(classifiers[c])):
            raise Exception('classifier not valid: ' + c)
    return True

def CheckKeys(Debug, included, excluded, price='',dataset='',classifier='',trader='',allocator=''):
    if price != '' and 'prices' in included and len(included['prices']) > 0 and price not in included['prices']:
        if Debug: print('price: ' + price + ' not in included: ' + str(included['prices']))
        return False
    if dataset != '' and 'datasets' in included and len(included['datasets']) > 0 and dataset not in included['datasets']:
        if Debug: print('dataset: ' + dataset + ' not in included: ' + str(included['datasets']))
        return False
    if classifier != '' and 'classifiers' in included and len(included['classifiers']) > 0 and classifier not in included['classifiers']:
        if Debug: print('classifier: ' + classifier + ' not in included: ' + str(included['classifiers']))
        return False
    if trader != '' and 'traders' in included and len(included['traders']) > 0 and trader not in included['traders']:
        if Debug: print('trader: ' + trader + ' not in included: ' + str(included['traders']))
        return False
    if allocator != '' and 'allocators' in included and len(included['allocators']) > 0 and allocator not in included['allocators']:
        if Debug: print('allocator: ' + allocator + ' not in included: ' + str(included['allocators']))
        return False
    
    if price != '' and 'prices' in excluded and len(excluded['prices']>0) and price in excluded['prices']:
        if Debug: print('price: ' + price + ' in excluded: ' + str(excluded['prices']))
        return False
    if dataset != '' and 'datasets' in excluded and len(excluded['datasets']) > 0 and dataset in excluded['datasets']:
        if Debug: print('dataset: ' + dataset + ' in included: ' + str(excluded['datasets']))
        return False
    if classifier != '' and 'classifiers' in excluded and len(excluded['classifiers']) > 0 and classifier in excluded['classifiers']:
        if Debug: print('classifier: ' + classifier + ' in excluded: ' + str(excluded['classifiers']))
        return False
    if trader != '' and 'traders' in excluded and len(excluded['traders']) > 0 and trader in excluded['traders']:
        if Debug: print('trader: ' + trader + ' in excluded: ' + str(excluded['traders']))
        return False
    if allocator != '' and 'allocators' in excluded and len(excluded['allocators']) > 0 and allocator in excluded['allocators']:
        if Debug: print('allocator: ' + allocator + ' in excluded: ' + str(excluded['allocators']))
        return False
    
    return True

def RunPriceData(runid, prices, start_date, end_date, run_mode='file_or_generate', save = True):
    price_data = {}
    for p in prices:
        if run_mode == 'file':
            print('loading price: ' + p)
            price_data[p] = pd.read_csv('prices\\' + p + '.csv', index_col = 'date')
        elif run_mode == 'generate':
            print('downloading price: ' + p)
            price_data[p] = prices2[p].CreatePriceData()      
        elif run_mode == 'file_or_generate':
            price_df = pd.read_csv('prices\\' + p + '.csv', index_col = 'date')
            ls_dtstr_in_file = price_df.index[-1]
            ls_dt_in_file = datetime.strptime(ls_dtstr_in_file, '%Y-%m-%d').date()
            edt = datetime.strptime(end_date, '%Y-%m-%d').date()
            if edt > ls_dt_in_file:
                print('downloading price: ' + p + ', required date is greater than last date in file: ' + ls_dtstr_in_file + ' < ' + end_date)
                price_data[p] = prices2[p].CreatePriceData()
            else:
                print('loading price: ' + p + ', file is up to date: ' + ls_dtstr_in_file + ' >= ' + end_date)
                price_data[p] = pd.read_csv('prices\\' + p + '.csv', index_col = 'date')

        if save and run_mode != 'file':
             price_data[p].to_csv('prices\\' + p + '.csv', index=True) 
    return price_data
    
def RunDatasetData(runid, prices, price_data, datasets, start_date, end_date, run_mode='file_or_generate', save = True):
    price_dataset_results = {}
    for p in prices:
        dataset_results = {}
        for d in datasets:
            if not CheckKeys(False, datasets[d].included, datasets[d].excluded, p):
                continue

            if run_mode == 'file':
                print('reading dataset: ' + d + '_' + p)
                dataset_results[d] = pd.read_csv('output\\' + runid + '\\datasets\\' + d + '_' + p + '.csv', index_col = 'date')
            if run_mode == 'generate':
                print('creating dataset for:' + d + '_' + p)
                dataset_results[d] = datasets[d].CreateDataset(p, price_data)
                 
            if save and run_mode != 'file':
                dataset_results[d].to_csv('output\\' + runid + '\\datasets\\' + d + '_' + p + '.csv', index=True)  

        price_dataset_results[p] = dataset_results
    return price_dataset_results
    
def RunClassifierData(runid, prices, price_data, datasets, price_dataset_data, classifiers, start_date, end_date, run_mode = 'file_or_generate', save = True):
    price_dataset_classifier_results = {}
    for p in prices:
        dataset_classifier_results = {}
        for d in datasets: 
            if not CheckKeys(False, datasets[d].included, datasets[d].excluded, p):
                continue
            classifier_results = {}
            for c in classifiers:
                if not CheckKeys(False, classifiers[c].included, classifiers[c].excluded, p, d):
                    continue
                if run_mode == 'file':
                    print('loading classifier data for: ' + c + '_' + d + '_' + p)
                    classifier_results[c] = pd.read_csv('output\\' + runid + '\\classifiers\\' + c + '_' + d + '_' + p + '.csv')
                elif run_mode == 'generate':
                    print('generating classifier data for: ' + c + '_' + d + '_' + p)
                    start_ind, end_ind = price_dataset_data[p][d].index.get_loc(start_date), price_dataset_data[p][d].index.get_loc(end_date)
                    classifier_results[c] = classifiers[c].GenerateModelResults(price_dataset_data[p][d], range(start_ind, end_ind)) 
                elif run_mode == 'file_or_generate':
                    print('loading or generating classifier data for: ' + c + '_' + d + '_' + p)
                    classifier_data = []
                    if os.path.exists('output\\' + runid + '\\classifiers\\' + c + '_' + d + '_' + p + '.csv'):
                        classifier_data.append(pd.read_csv('output\\' + runid + '\\classifiers\\' + c + '_' + d + '_' + p + '.csv', index_col = 'date'))
                    for run_range in classifiers[c].GetClassifierDatasetRunRange(start_date, end_date, price_dataset_data[p][d], classifier_data):
                        classifier_data.append(classifiers[c].GenerateModelResults(price_dataset_data[p][d], run_range))
                    classifier_results[c] = pd.concat(classifier_data)
                    classifier_results[c].sort_values(['date'], inplace=True)

                if save and run_mode != 'file':
                    classifier_results[c].to_csv('output\\' + runid + '\\classifiers\\' + c + '_' + d + '_' + p + '.csv', index=True)    

            dataset_classifier_results[d] = classifier_results
        price_dataset_classifier_results[p] = dataset_classifier_results
    return price_dataset_classifier_results
    
def RunTraderData(runid, prices, price_data, datasets, price_dataset_data, classifiers, price_dataset_classifier_data, traders, start_date, end_date, run_mode = 'generate', save = True):
    norm_price = True
    price_dataset_classifier_predictor_results = {}
    for p in prices:
        dataset_classifier_predictor_results = {}
        for d in datasets:
            if not CheckKeys(False, datasets[d].included, datasets[d].excluded, p):
                continue
            classifier_predictor_results = {}
            for c in classifiers:
                if not CheckKeys(False, classifiers[c].included, classifiers[c].excluded, p, d):
                    continue
                trader_results = {}
                for t in traders:
                    if not CheckKeys(False, traders[t].included, traders[t].excluded, p, d, c):
                        continue

                    if run_mode == 'file':
                        print('loading trader data for: ' + t + '_' + c + '_' + d + '_' + p)
                        trader_results[t] = pd.read_csv('output\\' + runid + '\\traders\\' + t + '_' + c + '_' + d + '_' + p + '.csv')
                    elif run_mode == 'generate':
                        print('generating trader data: ' + t + '_' + c + '_' + d + '_' + p)
                        start_ind, end_ind = price_dataset_classifier_data[p][d][c].index.get_loc(start_date), price_dataset_classifier_data[p][d][c].index.get_loc(end_date)
                        trader_results[t] = traders[t].GenerateTraderResults(price_data[p], price_dataset_classifier_data[p][d][c], norm_price, range(start_ind, end_ind))
                        trader_results[t]['price'] = p
                        trader_results[t]['dataset'] = d
                        trader_results[t]['classifier'] = c
                        trader_results[t]['predictor'] = t
                        trader_results[t]['exit_method'] = traders[t].exit_method
                        
                    if save and run_mode != 'file':
                        trader_results[t].to_csv('output\\' + runid + '\\traders\\' + t + '_' + c + '_' + d + '_' + p + '.csv', index=False)
                
                classifier_predictor_results[c] = trader_results
            dataset_classifier_predictor_results[d] = classifier_predictor_results
        price_dataset_classifier_predictor_results[p] = dataset_classifier_predictor_results
    return price_dataset_classifier_predictor_results

def RunAllocatorData(runid, prices, price_data, datasets, price_dataset_data, classifiers, price_dataset_classifier_data, traders, price_dataset_classifier_trader_data, allocators, start_date, end_date, run_mode = 'generate', save = True):
    allocator_results, allocator_returns = {}, {}
    for a in allocators:
        if run_mode == 'file':
            print('loading allocator data for: ' + a)
            allocator_results[a] = pd.read_csv('output\\' + runid + '\\allocators\\' + a + '.csv')
        elif run_mode == 'generate': # and a == 'A1':
            trader_dfs = []
            for p in prices:
                for d in datasets:
                    if not CheckKeys(False, datasets[d].included, datasets[d].excluded, p):
                        continue
                    for c in classifiers:
                        if not CheckKeys(False, classifiers[c].included, classifiers[c].excluded, p, d):
                            continue
                        for t in traders:
                            if not CheckKeys(False, traders[t].included, traders[t].excluded, p, d, c):
                                continue
                            if CheckKeys(False, allocators[a].included, allocators[a].excluded, p, d, c, t):
                                trader_dfs.append(price_dataset_classifier_trader_data[p][d][c][t])
            if len(trader_dfs)>0:
                print('generating allocator data for: ' + a)
                allocator_results[a] = allocators[a].GeneratePortfolioResults(trader_dfs)
                if save and run_mode != 'file':
                    allocator_results[a].to_csv('output\\' + runid + '\\allocators\\'  + a + '.csv', index=False)

        if a in allocator_results:
            print('generating return data for: ' + a)
            allocator_returns[a] = allocators[a].GeneratePortfolioReturns(allocator_results[a])
    return allocator_results, allocator_returns
            