import numpy as np
import pandas as pd
from ta import add_all_ta_features

class Technical_Diffshift:
    def __init__(self, symbols, pivot_cols, pivots, technical_cols, peak_cols = ['open'], label_col = 'close', included = {}, excluded = {}):
        #self.label = label
        self.symbols = symbols
        self.pivot_cols = pivot_cols
        self.pivots = pivots
        self.technical_cols = technical_cols
        self.peak_cols = peak_cols
        self.label_col = label_col
        self.included = included
        self.excluded = excluded
    
    def CreateDataset(self, label_symbol, stock_dict, vol_period = 20, target_std = 0.015):
        full_res, label_df, base_prices = [], None, None
        for symbol in self.symbols: 
            sym_dfs = []
            df = stock_dict[symbol]
            #print(df)
            sym_dfs.append(self.DiffShift(df, self.pivot_cols, self.pivots)) # pivot
            sym_dfs.append(self.DiffShift(df, self.peak_cols, {1:[0]})) # peak ahead
            technicals = self.CalculateTechnicals(df, self.technical_cols) # calculate technicals
            #print(technicals)
            sym_dfs.append(self.DiffShift(technicals, self.technical_cols, {0:[1]})) #shifted technicals
            sym_df = pd.concat(sym_dfs, axis=1, join="inner").add_prefix(symbol + '_') #combined symbol data
            full_res.append(sym_df)
            
            if symbol == label_symbol:
                label_df = df
                label_df['label'] = np.where(label_df['close']>label_df['open'], 1, 0)
                label_df = label_df['label']
                base_prices = df[['open','high','low','close']]
                
        full_res.append(label_df)
        res_df = pd.concat(full_res, axis=1, join='inner')
        #base_prices = base_prices[base_prices.index.isin(res_df.index)]
        return res_df #, base_prices

    def DiffShift(self, df, columns, diffs):
        final_dfs = []
        for diff in diffs.keys():
            diff_df = pd.DataFrame()   
            for col in columns:
                if diff == 0:
                    diff_df[col] = df[col]
                else:
                    diff_df[col+'_D'+str(diff)] = df[col].rolling(window=diff+1).apply(lambda x: x.iloc[diff] - x.iloc[0])
            sampled_dfs = []
            for shift in diffs[diff]:
                sampled_df = diff_df.shift(periods=shift)
                sampled_df = sampled_df.add_suffix('_S'+str(shift))
                sampled_dfs.append(sampled_df)
            final_df = pd.concat(sampled_dfs, axis=1, join="inner")
            final_dfs.append(final_df)   
        pivot_df = pd.concat(final_dfs, axis=1, join="inner")
        return pivot_df
        
    def CalculateTechnicals(self, df, technical_cols):
        cols = ['open', 'high', 'low', 'close', 'volume']
        technicals = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume")
        technicals = technicals[technical_cols]
        return technicals
