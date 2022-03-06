import numpy as np
import pandas as pd
import requests

AV_KEY='5AK7ZPDAGCNO39B7'

class AVPrice:
    def __init__(self, symbol, normalise=True, std_period = 20, target_std = 0.015):
        self.symbol = symbol
        self.normalise = normalise
        self.std_period = std_period
        self.target_std = target_std
        
    def CreatePriceData(self): 
        cols = ['open', 'high', 'low', 'close', 'volume']
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+self.symbol+'&outputsize=full&apikey='+AV_KEY
        print(url)
        r = ''
        try:
            r = requests.get(url)
            data = r.json()
            dic = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(dic, orient='index')
            df = df[['1. open', '2. high', '3. low', '4. close', '5. volume']]
            df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low','4. close':'close', '5. volume':'volume'})
            df.index.name = 'date'
            df = df.sort_index(ascending = True)
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            
            if self.normalise:
                df = self.CreateNormalisedPrice(df)
            
            return df
        except Exception as e: 
            print(e)
            print(r)
        return None
        
    def CreateNormalisedPrice(self, base_prices):
        base_prices['open_1D_diff_perc'] = base_prices['open'].rolling(window=2).apply(lambda x: (x.iloc[1] - x.iloc[0])/x.iloc[0])
        base_prices['open_1D_diff_perc_'+str(self.std_period)+'D_STD'] = base_prices['open_1D_diff_perc'].rolling(self.std_period).std()
        base_prices['target_STD'] = self.target_std
        base_prices['leverage'] = base_prices['target_STD'] / base_prices['open_1D_diff_perc_'+str(self.std_period)+'D_STD']
        base_prices['norm_open'] = base_prices['open'] * base_prices['leverage']
        base_prices['norm_high'] = base_prices['high'] * base_prices['leverage']
        base_prices['norm_low'] = base_prices['low'] * base_prices['leverage']
        base_prices['norm_close'] = base_prices['close'] * base_prices['leverage']
        return base_prices
        
