import pandas as pd
import numpy as np
import random

# Auxilary function for question #5
def binning(col, cut_points, labels=None):
    # Define min and max values:
    minval = col.min()
    maxval = col.max()

    # create list by adding min and max to cut_points
    break_points = [minval] + cut_points + [maxval]

    # if no labels provided, use default labels 0 ... (n-1)
    if not labels:
        labels = range(len(cut_points)+1)

    # Binning using cut function of pandas
    colBin = pd.cut(col, bins=break_points, labels=labels, include_lowest=True)
    return colBin


print("\n --- Question #1 ---")
print("\n Loading train_Loan.csv")
df = pd.read_csv("/home/serfati/Desktop/DS/Labs/lab_9/train_Loan.csv")             # Reading the dataset in a dataframe using Pandas

print("\n --- Question #2 ---")
keys=list(df.keys())
for k in keys[1:]:
    print("\n Frequency Distribution of " + str(k) + " attribute:")
    print(df[str(k)].value_counts())                        # Frequency distribution for each attributes

print("\n --- Question #3 ---")
# print the data types of the attributes in the DataFrame
print("\n The data types of the attributes in the DataFrame:")
print(df.dtypes)

print("\n --- Question #4 ---")
print("\n Data-Frame BEFORE Missing-values correction:")
print(df)
g=list(df['Gender'].dropna().unique())
df['Gender'].fillna(random.choice(g), inplace=True)
m=list(df['Married'].dropna().unique())
df['Married'].fillna(random.choice(m), inplace=True)
se=list(df['Self_Employed'].dropna().unique())
df['Self_Employed'].fillna(random.choice(se), inplace=True)
print("\n Data-Frame AFTER Missing-values correction:")
print(df)

print("\n --- Question #5 ---")
# Define bins as 0<=x<100, 100<=x<200, 200<=x<300, x>=300
bins = [100, 200, 300]
group_names = ['Low', 'Medium', 'High', 'Extreme']
# Discretize the values in LoanAmount attribute
df["LoanAmount_Bin"] = binning(df["LoanAmount"], bins, group_names)
# Count the number of observations which each value
print(pd.value_counts(df["LoanAmount_Bin"], sort=False))
print(df)

print("\n --- Question #6 ---")
# Keep only the ones that are within +3 to -3 standard deviations in the column 'LoanAmount'.
print(df[(np.abs(df["LoanAmount"]-df["LoanAmount"].mean()) <= (3*df["LoanAmount"].std()))])
