import math
import numpy as np
import pandas as pd

class Mean_LERP:
    def __init__(self, mean_periods = 5, min_lerp = 0, max_lerp = 1, included = {}, excluded = {}):
        self.mean_periods = mean_periods
        self.min_lerp = min_lerp
        self.max_lerp = max_lerp
        self.included = included
        self.excluded = excluded
        
    def Calc_Predictor_Weights(self, predictors):
        tot_series = None
        mean_cl, mean_cl_s = str(self.mean_periods)+'D_mean_pal', str(self.mean_periods)+'D_mean_pal_S1'
        for pred in predictors:
            #pred['min'], pred['max'] = 0.0, 0.0
            #pred[mean_cl] = pred['perc_pal'].rolling(self.mean_periods).mean()
            pred[mean_cl_s] = pred['perc_pal'].rolling(self.mean_periods).mean().shift(1)
            pred['lerp'] = 1.0
            pred['weight'] = 1.0
            
        for i in range(0, len(predictors[0].index)):
            minval, maxval, tot = 10000, -10000, 0.0
            for pred in predictors:
                val = pred.iloc[i][mean_cl_s]
                if val < minval:
                    minval = val
                if val > maxval:
                    maxval = val
                
            for pred in predictors:
                #if minval < 10000:
                #    pred.at[i, 'min'] = minval
                #if maxval > -10000:
                #    pred.at[i, 'max'] = maxval
                val = pred.iloc[i][mean_cl_s]
                lerp = self.min_lerp + (val-minval) / (maxval-minval) * (self.max_lerp - self.min_lerp)
                tot += lerp
                #print('i: ' + str(i) + ', min: ' + str(minval) + ', max: ' + str(maxval) + ', val: ' + str(val) + ', lerp: ' + str(lerp) + ', tot: ' + str(tot) + ', weight: ' + str(lerp/tot))
                if not math.isnan(lerp):
                    pred.at[i, 'lerp'] = lerp
                #print(pred)
            
            for pred in predictors:
                lerp = pred.at[i, 'lerp']
                if not math.isnan(tot) and lerp > 0.0:
                    pred.at[i, 'weight'] = lerp/tot
        return predictors
    
    def GeneratePortfolioResults(self, symbol_classifier_predictor_results):
        frames = symbol_classifier_predictor_results
        if len(frames) > 0:       
            frames = self.Calc_Predictor_Weights(frames)
            port_df = pd.concat(frames)
            port_df['weighted_pal'] = port_df['perc_pal'] * port_df['weight']
            port_df['amount'] = port_df['weight'] * port_df['prediction']
            port_df = port_df[['date','price','dataset','classifier','predictor','exit_method','bought_at','prediction','5D_mean_pal_S1','lerp','weight','amount','sold_at','perc_pal','weighted_pal']]
            return port_df
        else:
            print('no traders included in allocator')
            return None
    
    def GeneratePortfolioReturns(self, portfolio_result, rolls = [3, 5]):
        df = portfolio_result[['date','weight','weighted_pal']]
        df = df.groupby(['date']).sum().reset_index()
        for roll in rolls:
            df[str(roll) + 'D_mean'] = df['weighted_pal'].rolling(roll).mean()
            df[str(roll) + 'D_cumprod'] = (1 + df['weighted_pal']).rolling(roll).apply(np.prod, raw=True) - 1
        return df
    
    def GenerateInstructions(self, date, portfolio_result):
        df = portfolio_result[(portfolio_result['date'] == date)]
        df = df[['date','dataset','exit_method','bought_at','amount']]
        df = df.groupby(['date','dataset','exit_method','bought_at']).sum().reset_index()
        return df
