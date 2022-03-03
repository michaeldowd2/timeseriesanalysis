import pandas as pd

class Binary_Trader:
    def __init__(self, threshold, exit_method, capture_thresh = 0.01, trade_longs = True, trade_shorts = True, included = {}, excluded = {}):
        self.exit_method = exit_method
        self.threshold = threshold
        self.capture_thresh = capture_thresh
        self.trade_longs = trade_longs
        self.trade_shorts = trade_shorts
        self.included = included
        self.excluded = excluded

    def get_prediction(self, index, predictions):
        pred = predictions.iloc[index]['prediction']
        if pred > 0 and self.trade_longs:
            return 1
        elif pred < 0 and self.trade_shorts:
            return -1
        return 0

    def exit_at(self, prediction, open, close, high, low):
        if self.exit_method == 'at_close':
            return close
        elif self.exit_method == 'capture_gains':
            if prediction < 0 and low <= open - open * self.capture_thresh:
                return open - open * self.capture_thresh
            elif prediction > 0 and high >= open + open * self.capture_thresh:
                return open + open * self.capture_thresh
            else:
                return close
            
    def percent_pal(self, prediction, open, close, high, low):
        if prediction == 0:
            return 0
        else:
            bet_dir = prediction
            exit_price = self.exit_at(prediction, open, close, high, low)
            return (exit_price - open) / open * bet_dir
    
    def GenerateTraderResults(self, base_prices, classifier_results, norm_price, run_range):
        res_dict = {'date':[],'bought_at':[], 'prediction':[], 'sold_at':[], 'perc_pal':[]}
        
        for i in range(run_range[0], run_range[-1]+2):
            date = classifier_results.index[i]
            price_ind = base_prices.index.get_loc(date)
            price_ind = base_prices.index.get_loc(date)
            
            open = base_prices.iloc[price_ind]['open']
            close = base_prices.iloc[price_ind]['close']
            high = base_prices.iloc[price_ind]['high']
            low = base_prices.iloc[price_ind]['low']
            if norm_price:
                open = base_prices.iloc[price_ind]['norm_open']
                close = base_prices.iloc[price_ind]['norm_close']
                high = base_prices.iloc[price_ind]['norm_high']
                low = base_prices.iloc[price_ind]['norm_low']

            prediction = self.get_prediction(i, classifier_results)
            res_dict['date'].append(date)
            res_dict['bought_at'].append(open)
            res_dict['prediction'].append(prediction)
            res_dict['sold_at'].append(self.exit_at(prediction, open, close, high, low))
            res_dict['perc_pal'].append(self.percent_pal(prediction, open, close, high, low))
        return pd.DataFrame(res_dict)