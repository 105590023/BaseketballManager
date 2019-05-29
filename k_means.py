#%%
import numpy as np 
import matplotlib.pyplot as plt  

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# 讀取測試資料
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

# 列出輸入資料
print("***** Train_Set *****")
print(train.describe())
print("\n")
print("***** Test_Set *****")
print(test.describe())

# 列出資料標頭
print(train.columns.values)

# 列出輸入資料缺失總數
print("*****In the train set*****")
print(train.isna().sum())
print("\n")
print("*****In the test set*****")
print(test.isna().sum())

# 以平均值修補缺失值
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

print("*****In the train set*****")
print(train.isna().sum())
print("\n")
print("*****In the test set*****")
print(test.isna().sum())

#%%
train['Ticket'].head()
train['Cabin'].head()

#%%
# 關於Pclass的生存計數
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#%%
# 關於性別的生存計數
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#%%
# 關於SibSp的生存計數
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#%%
# 年齡與倖存者”的關係圖
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)

#%%
# Pclass和Survived功能如何與圖表相互關聯
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

#%%
# 數據類型
train.info()

#%%
# 刪除功能
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

#%%
# 將“性別”特徵轉換為數字特徵
labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

#%%
# 訓練K-Means模型
# drop()功能從數據中刪除Survival列
X = np.array(train.drop(['Survived'], 1).astype(float))
y = np.array(train['Survived'])

# 構建K-Means模型
kmeans = KMeans(n_clusters=2) # You want cluster the passenger records into 2: Survived or Not survived

# 把0 - 1作為所有特徵的統一值範圍
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans.fit(X_scaled)

#%%
# 模型的準確性
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))