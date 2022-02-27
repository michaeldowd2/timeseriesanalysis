class Binary_Trader:
    def __init__(self, threshold, exit_method, capture_thresh = 0.01, trade_longs = True, trade_shorts = True, tags = [], included = {}, excluded = {}):
        self.exit_method = exit_method
        self.threshold = threshold
        self.capture_thresh = capture_thresh
        self.trade_longs = trade_longs
        self.trade_shorts = trade_shorts
        self.tags = tags
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