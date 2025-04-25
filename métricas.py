from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, confusion_matrix

def metricas(y_ground, y_hat):
    results = {'Accuracy': accuracy_score(y_ground, y_hat),
               'Precision':precision_score(y_ground, y_hat, average = 'micro'),
               'Recall': recall_score(y_ground, y_hat, average = 'micro'),
               'f1_score': f1_score(y_ground, y_hat, average = 'micro'),
               'balanced_accuracy': balanced_accuracy_score(y_ground, y_hat),
               'AUC': roc_auc_score(y_ground, y_hat)}
    return(results)