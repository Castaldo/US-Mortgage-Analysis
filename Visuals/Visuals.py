import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import matplotlib.axes
import math

Mortgage = pd.read_csv('Data/Processed/Mortgage.csv')

sns.set(style="dark")
plt.figure(figsize=(8,8))
ax = sns.countplot(x="Loan_Accepted", data=Mortgage, palette = "pastel")
ax.set(xlabel='Loan Accepted', ylabel='Count')
Mortgage["Loan_Accepted"].value_counts(normalize = 'index')

plt.savefig('Visuals/Plots/LoanAcceptance.png')
print('Loan Acceptance Chart Saved')

sns.set(style="dark")
plt.figure(figsize=(30,20))
ax = sns.countplot(x="state_name", data=Mortgage)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()

plt.savefig('Visuals/Plots/MortgagesByState.png')
print('Mortgages By State Chart Saved')

Mortgage.loc[Mortgage['applicant_sex_name'].str.contains('Information not')] = 'Unknown'

sns.set(style="dark")
plt.figure(figsize=(8,5))
ax = sns.countplot(x="applicant_sex_name", data=Mortgage, palette = "pastel")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()

plt.savefig('Visuals/Plots/ApplicantGender.png')
print('Applicants by Gender Chart Saved')

Mortgage.loc[Mortgage['applicant_race_name_1'].str.contains('Information not')] = 'Unknown'

sns.set(style="dark")
plt.figure(figsize=(8,8))
ax = sns.countplot(x="applicant_race_name_1", data=Mortgage, palette = "pastel")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()

plt.savefig('Visuals/Plots/ApplicantEthnicity.png')
print('Applicants by Ethnicity Chart Saved')

Mortgage = Mortgage[Mortgage.loan_type != 'Unknown']

plt.figure(figsize=(10,8))
plt.scatter(Mortgage['applicant_income_000s'], Mortgage['loan_amount_000s'], c = Mortgage['Loan_Accepted'])
plt.xlabel('Applicant Income in 000s')
plt.ylabel('Loan Amount in 000s')
plt.colorbar()

plt.savefig('Visuals/Plots/IncomeLoanRatio.png')
print('Income to Loan Ratio Chart Saved')

Mortgage.drop(Mortgage.columns[[6, 7, 9, 14, 17]], axis=1, inplace=True)
to_log = ["loan_amount_000s", "applicant_income_000s"]
Mortgage[to_log] = Mortgage[to_log].applymap(math.log)

Cols = ["loan_amount_000s", "applicant_income_000s", "population", "minority_population", "tract_to_msamd_income", 
        "hud_median_family_income","number_of_owner_occupied_units", "number_of_1_to_4_family_units", "msamd", 
        "county_code"]

fig, axes = plt.subplots(ncols = 2, nrows = 5, figsize = (15,18))
fig.subplots_adjust(hspace = 0.5, wspace = 0.3)

for ax, col in zip(axes.flatten(), Cols) :
    sns.kdeplot(Mortgage[Mortgage["Loan_Accepted"] == 0][col], shade="True", label="Not accepted", ax = ax)
    sns.kdeplot(Mortgage[Mortgage["Loan_Accepted"] == 1][col], shade="True", label="Accepted", ax = ax)
    ax.set_xlabel(col)

plt.savefig('Visuals/Plots/Kernal.png')
print('Kernal Chart Saved')

Mortgage = Mortgage.astype('float64')
plt.figure(figsize=(16,12))
sns.heatmap(Mortgage.corr().round(decimals=2), annot=True)
plt.title("Correlation heatmap")

plt.savefig('Visuals/Plots/CorrelationMatrix.png')
print('Correlation Matrix Saved')
