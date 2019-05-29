#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# 讀取測試資料
playerData = pd.read_csv("./csv_data.csv")

# 列出輸入資料
# print("------- playerData -------")
# print(playerData.describe())
# print(playerData.head())
# print("\n")

# 列出資料標頭
# print(playerData.columns.values)

# 查看資料類型
# playerData.info()

#%%
# 訓練K-Means模型
data = playerData[['PTS', 'TRB', 'AST', 'STL', 'BLK']]
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(data)

# 構建K-Means模型
kmeans = KMeans(n_clusters=5)

# 把0 - 1作為所有特徵的統一值範圍
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(reduced_X)

kmeans.fit(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_)
plt.show()