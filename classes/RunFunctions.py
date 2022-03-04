import os
import pandas as pd
from datetime import datetime

from classes.prices.AVPrice import AVPrice
from classes.datasets.Technical_Diffshift import Technical_Diffshift

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

