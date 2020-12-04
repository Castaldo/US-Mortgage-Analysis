import pandas as pd

Mortgage = pd.read_csv('Data/Unprocessed/hmda_2017_nationwide_all-records_labels.csv')

Mortgage = Mortgage[(Mortgage['loan_purpose_name'] == 'Refinancing')]
Mortgage = Mortgage[(Mortgage['action_taken'] <= 3)]

Mortgage.drop(Mortgage.columns[[0, 1, 2, 3, 5, 7, 9, 10, 11, 14, 15, 16, 18, 26, 28, 32, 33, 34, 35, 36, 37,
                        38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 52, 55, 57, 58, 59, 60, 61, 62, 63, 64,
                        65, 66, 68, 69, 70, 77]], axis=1, inplace=True)

def f(row):
    if row['action_taken'] < 3:
        val = 1
    else:
        val = 0
    return val

Mortgage['Loan_Accepted'] = Mortgage.apply(f, axis=1)
del Mortgage['action_taken']

Mortgage.to_csv(r'Data/Processed/Mortgage.csv', index=False)
print('Mortgage Data Processed')

