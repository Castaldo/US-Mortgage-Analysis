import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report 
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score

def Clean(data):
    data = pd.read_csv('Data/Processed/Mortgage.csv')
    data.drop(data.columns[[6, 7, 9, 14, 17, 2, 3, 5, 6, 7, 8, 12, 16, 17, 22, 23]], axis=1, inplace=True)
    data = data.dropna()
    return data

def mod_eval(df ,predictions, predprob, y_test, title):
      
    cm = confusion_matrix(df.Loan_Accepted[y_test.index], predictions)
    sns.heatmap(cm, annot = True, fmt = '.5g', xticklabels = ['No', 'Yes'], yticklabels = ['No', 'Yes']).set_title(title)
    plt.xlabel('Real Values') 
    plt.ylabel('Predicted Values')
    print(classification_report(df.Loan_Accepted[y_test.index], predictions))
    
    f, axes = plt.subplots(1,2,figsize= (20,6),squeeze=False)
    
    false_positives, true_positives, _ = roc_curve(df.Loan_Accepted[y_test.index], predprob[:,1])
    roc_auc = auc(false_positives, true_positives)
    
    axes[0,0].plot(false_positives, true_positives, lw = 3)
    axes[0,0].set_title('{} ROC curve (area = {:0.2f})'.format(title, roc_auc))
    axes[0,0].set(xlabel='False Positive %',ylabel='True Positive %')
    axes[0,0].grid(b=True, which='both', axis='both', color='grey', linestyle = '-', linewidth = '0.5')

    precision, recall, thresholds = precision_recall_curve(y_test, predprob[:,1])
    best_index = np.argmin(np.abs(precision-recall))
    
    axes[0,1].plot(precision,recall)
    axes[0,1].set_title('{} Precision-Recall Curve'.format(title))
    axes[0,1].set(xlabel='Precision', ylabel='Recall', xlim=(0.4,1.05))
    axes[0,1].plot(precision[best_index],recall[best_index],'o',color='r')
    axes[0,1].grid(b=True, which='both', axis='both', color='grey', linestyle = '-', linewidth = '0.5')

def model_training(classifier,df):
    clf = classifier
    t = df.drop(columns=['Loan_Accepted'])
    X_train, X_test, y_train, y_tests = train_test_split(t, df['Loan_Accepted'], test_size = .33, stratify = df['Loan_Accepted'])
    clf.fit(X_train, y_train)
    return clf

