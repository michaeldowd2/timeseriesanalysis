import math
class Mean_LERP_Portfolio:
    def __init__(self, mean_periods = 5, min_lerp = 0, max_lerp = 1):
        self.mean_periods = mean_periods
        self.min_lerp = min_lerp
        self.max_lerp = max_lerp
        
    def Calc_Predictor_Weights(self, predictors):
        tot_series = None
        mean_cl, mean_cl_s = str(self.mean_periods)+'D_mean_pal', str(self.mean_periods)+'D_mean_pal_S1'
        for pred in predictors:
            #pred['min'], pred['max'] = 0.0, 0.0
            #pred[mean_cl] = pred['perc_pal'].rolling(self.mean_periods).mean()
            pred[mean_cl_s] = pred['perc_pal'].rolling(self.mean_periods).mean().shift(1)
            pred['lerp'] = 0.0
            pred['weight'] = 0.0
            
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
                #print('i: ' + str(i) + ', min: ' + str(minval) + ', max: ' + str(maxval) + ', val: ' + str(val) + ', weight: ' + str(weight))
                if not math.isnan(lerp):
                    pred.at[i, 'lerp'] = lerp
                #print(pred)
            
            for pred in predictors:
                lerp = pred.at[i, 'lerp']
                if lerp > 0.0:
                    pred.at[i, 'weight'] = lerp/tot
        return predictors