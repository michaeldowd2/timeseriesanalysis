class Binary_Predictor:
    def __init__(self, threshold, exit_method, models, pred_method = 'ave', capture_thresh = 0.01):
        self.exit_method = exit_method
        self.threshold = threshold
        self.models = models
        self.pred_method = pred_method
        self.capture_thresh = capture_thresh

    def get_prediction(self, index, predictions):
        return predictions.iloc[index]['prediction']

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
        bet_dir = prediction
        exit_price = self.exit_at(prediction, open, close, high, low)
        return (exit_price - open) / open * bet_dir