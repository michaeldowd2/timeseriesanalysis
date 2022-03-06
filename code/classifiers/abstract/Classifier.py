import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Classifier:
    def CheckClass(self, subclass):
        sub = issubclass(subclass, type(self))
        if not sub:
            return False
        has_method = (hasattr(subclass, 'GenerateModelResults') and callable(subclass.GenerateModelResults))
        if not has_method:
            return False
        #function params and return
        args = inspect.getargspec(subclass.GenerateModelResults).args
        if 'self' not in args: return False
        if 'dataset' not in args: return False
        if 'run_range' not in args: return False
        return True
        
    def SampleDataframe(self, df, date, no_samples):
        ind = df.index.get_loc(date)
        if ind-no_samples > 0:
            df = df.iloc[ind+1-no_samples:ind+1,:]
            return df
        return None
    
    def PlotPredVsAct(self, pred, acts, title):
        plt.figure(figsize=(7, 2))
        plt.plot(pred, label = "predictions")
        plt.plot(acts, label = "actual")
        plt.legend(loc="upper left")
        
        plt.title(title)
        plt.show()
        
    def GetClassifierDatasetRunRange(self, start_date, end_date, dataset, classifier_data):
        run_ranges = []
        try:
            start_ind, end_ind = dataset.index.get_loc(start_date), dataset.index.get_loc(end_date)
        except:
            print('dates not found in underlying dataset, aborting generating classifier data')
            return
        if start_ind > end_ind:
            print('start_date is not earlier than end date, aborting generating classifier data')
        elif len(classifier_data) == 0:
            print('no file, running for full range: ' + start_date + ' - ' + end_date)
            run_ranges.append(range(start_ind, end_ind))
        else:
            existing_data = classifier_data[0]
            existing_sdate, existing_edate = existing_data.index[0], existing_data.index[-1]
            existing_start_ind, existing_end_ind = dataset.index.get_loc(existing_sdate), dataset.index.get_loc(existing_edate)
            if end_ind <= existing_end_ind and start_ind >= existing_start_ind:
                print('date range exists fully in file, not running any range')
            elif end_ind < existing_start_ind:
                print('existing file starts after specified end date - running up to start of file: ' + start_date + ' -> ' + existing_sdate)
                run_ranges.append(range(start_ind, existing_start_ind))
            elif start_ind > existing_end_ind:
                print('existing file ends before specified start date - running from end of file: ' + existing_edate + ' -> ' + end_date)
                run_ranges.append(range(existing_end_ind, end_ind))
            else: 
                if start_ind < existing_start_ind:
                    print('running up to start of file: ' + start_date + ' -> ' + existing_sdate)
                    run_ranges.append(range(start_ind, existing_start_ind))
                if end_ind > existing_end_ind+1:
                    print('running from end of file: ' + existing_edate + ' -> ' + end_date)
                    run_ranges.append(range(existing_end_ind, end_ind))
        return run_ranges 
        
