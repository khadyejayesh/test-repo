import pandas as pd

# ------------------------------
# Step 1: Input quarterly sales data
# ------------------------------
data = {
    'Year': [2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023],
    'Quarter': ['Q1','Q2','Q3','Q4','Q1','Q2','Q3','Q4'],
    'Sales': [120, 150, 180, 210, 140, 175, 210, 245]
}

df = pd.DataFrame(data)

print("Original Data:\n", df)

# ------------------------------
# Step 2: Calculate Link Relatives
# ------------------------------
df['Link_Relative'] = (df['Sales'] / df['Sales'].shift(1)) * 100

print("\nLink Relatives:\n", df)

# ------------------------------
# Step 3: Average Link Relatives by Quarter
# ------------------------------
avg_lr = df.groupby('Quarter')['Link_Relative'].mean()

print("\nAverage Link Relatives:\n", avg_lr)

# ------------------------------
# Step 4: Calculate Chain Relatives
# ------------------------------
chain_relatives = {}
chain_relatives['Q1'] = 100

quarters = ['Q1', 'Q2', 'Q3', 'Q4']

for i in range(1, len(quarters)):
    prev_q = quarters[i-1]
    curr_q = quarters[i]
    chain_relatives[curr_q] = (chain_relatives[prev_q] * avg_lr[curr_q]) / 100

cr_df = pd.DataFrame.from_dict(chain_relatives, orient='index',
                               columns=['Chain_Relative'])

print("\nChain Relatives:\n", cr_df)

# ------------------------------
# Step 5: Apply Correction Factor
# ------------------------------
correction_factor = 400 / cr_df['Chain_Relative'].sum()

cr_df['Seasonal_Index'] = cr_df['Chain_Relative'] * correction_factor

print("\nFinal Seasonal Indices:\n", cr_df)