import pickle, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

os.chdir(r'c:\딥러닝프로젝트')

df = pd.read_csv('data/size_dataset.csv', encoding='utf-8-sig').dropna()
X = df[['height', 'weight']].values
y = df['size'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_test_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

with open('model/model_b_knn.pkl', 'rb') as f:
    d = pickle.load(f)
knn = d['knn']
scaler = d['scaler']
X_test_scaled = scaler.transform(X_test)

y_pred = knn.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f'정확도 (Exact): {acc:.4f} ({acc:.2%})')

# 인접 허용 정확도 (±1 사이즈)
SIZE_ORDER = ['XS','S','M','L','XL','2XL']
def adj_acc(yt, yp):
    hit = 0
    for t, p in zip(yt, yp):
        ti = SIZE_ORDER.index(t) if t in SIZE_ORDER else -1
        pi = SIZE_ORDER.index(p) if p in SIZE_ORDER else -1
        if abs(ti - pi) <= 1:
            hit += 1
    return hit / len(yt)

print(f'정확도 (±1 허용): {adj_acc(y_test, y_pred):.4f} ({adj_acc(y_test, y_pred):.2%})')
print()
print(classification_report(y_test, y_pred, labels=['XS','S','M','L','XL']))
