import Pre_Processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

lr = Pre_Processing.Clean('Data/Processed/Mortgage.csv')
linear_regression = Pre_Processing.model_training(LogisticRegression(C =.01,  penalty = 'l1', solver='liblinear', max_iter=1000) , lr)

t = lr.drop(columns=['Loan_Accepted'])
X_train, X_test, y_train, y_test = Pre_Processing.train_test_split(t, lr['Loan_Accepted'], test_size=0.5, random_state = 42, 
                                                    stratify= lr['Loan_Accepted'])

t = .5
predprob = lr.predict_proba(X_test)

pred_y = [np.ceil(x) if x >= t else np.floor(x) for x in predprob[:,1]]

Pre_Processing.mod_eval(lr, pred_y, lr.predict_proba(X_test), y_test, 'Logistic Regressin') 
