# %matplotlib inline
# import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt

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
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)