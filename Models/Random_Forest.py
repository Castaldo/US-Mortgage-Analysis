import Pre_Processing
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

RF = Pre_Processing.Clean('Data/Processed/Mortgage.csv')
random_forest = Pre_Processing.model_training(RandomForestClassifier(random_state = 42, n_jobs = 4, n_estimators = 100, max_depth = 14), RF)

t = RF.drop(columns=['Loan_Accepted'])
X_train, X_test, y_train, y_test = Pre_Processing.train_test_split(t, RF['Loan_Accepted'], test_size = .5, 
                                                    stratify = RF['Loan_Accepted'])

Pre_Processing.mod_eval(RF, random_forest.predict(X_test), random_forest.predict_proba(X_test), y_test, 'Random Forest')
