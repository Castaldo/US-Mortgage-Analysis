import Pre_Processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB

Gaussian = Pre_Processing.Clean('Data/Processed/Mortgage.csv')
gnb = Pre_Processing.model_training(GaussianNB(), Gaussian)

t = Gaussian.drop(columns=['Loan_Accepted'])
X_train, X_test, y_train, y_test = Pre_Processing.train_test_split(t , Gaussian['Loan_Accepted'], test_size=.5, random_state = 42, 
                                                    stratify = Gaussian['Loan_Accepted'])


t = .5
predprob = gnb.predict_proba(X_test)

pred_y = [np.ceil(x) if x >= t else np.floor(x) for x in predprob[:,1]]
Pre_Processing.mod_eval(Gaussian, pred_y, gnb.predict_proba(X_test), y_test, 'GaussianNB')
