import pandas as pd
import os
os.chdir(r'c:\딥러닝프로젝트')
df = pd.read_csv('data/size_dataset.csv', encoding='utf-8-sig')
print(df.shape)
print(df.columns.tolist())
print(df.head(5).to_string())
print('\nsize 분포:')
print(df['size'].value_counts().to_string())
