import pandas as pd
df= pd.read_csv('/home/agnes/desktop/flib/AMLsim/outputs/simtest/tx_log.csv')

print(df.head())

for col in df.columns:
    print(col)