import numpy as np
import pandas as pd
from random import seed
from random import random

class Benchmark:
    def __init__(self, benchmark_type, included = {}, excluded = {}):
        seed(42)
        self.benchmark_type = benchmark_type
        self.included = included
        self.excluded = excluded
        
    def GenerateModelResults(self, dataset, base_prices, run_range):
        res_dict = {'date':[], 'params':[], 'test_F1': [], 'prediction':[]}
        for i in range(run_range[0], run_range[-1]+2):
            date = dataset.index[i]
            prev_date = dataset.index[i-1]
            
            res_dict['date'].append(date)
            
            prediction = -1
            if self.benchmark_type == 'long':
                prediction = 1
            if self.benchmark_type == 'random' and random() > 0.5:
                prediction = 1
                
            res_dict['params'].append('Benchmark: ' + self.benchmark_type)
            res_dict['test_F1'].append(0)
            res_dict['prediction'].append(prediction)

        res = pd.DataFrame(res_dict).set_index('date')
        return res
