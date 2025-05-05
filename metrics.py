# metrics framework
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# calc metrics used for eda
def calc_metrics(self, y, y_pred):
    '''
    Function arguments include:
        y -> true y values
        y_pred -> predicted y values by the model
    '''
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    # return the metrics
    return acc, precision, recall, f1, mcc