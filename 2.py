import pandas as pd
import numpy as np

print("\n --- Question #1 ---")
print("\n Loading train_Loan.csv")
df = pd.read_csv("path-to/train_Loan.csv")

print("\n --- Question #2 ---")
print("\n Adding new column 'NormalizedIncome' using numpy and applying appropriate lambda function")


def half_square_root(x): return 0.5 * np.sqrt(x)


df['NormalizedIncome'] = df['ApplicantIncome'].apply(half_square_root)

print("\n --- Question #3 ---")
print("\n Adding new column 'Graduate' which holds '1' for Education='Graduate' "
      "and '0' for Education='Not Graduate' using pandas' dummy variable")
df_Education = pd.get_dummies(df['Education'])
df = df.join(df_Education['Graduate'])

