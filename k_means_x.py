#%%
import csv
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

_player = [0] * 5
print("請輸入球員資訊：")
_player[0] = input("得分：")
_player[1] = input("籃板：")
_player[2] = input("助攻：")
_player[3] = input("抄截：")
_player[4] = input("火鍋：")

headers = ['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

# 將輸入資料加在csv_data後面
with open('csv_data.csv', 'a', newline = '') as f:
    writer = csv.writer(f)
    # writer.writerow(headers)
    data = [('0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', _player[1],  _player[2],  _player[3],  _player[4], '0', '0', _player[0])]
    writer.writerows(data)



# 讀取測試資料
playerData = pd.read_csv("./csv_data.csv")

# playerData = pd.read_csv("./player.csv")

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
# test_X = pca.fit_transform(_player)gㄕ

# 構建K-Means模型
kmeans = KMeans(n_clusters=5)

# 把0 - 1作為所有特徵的統一值範圍
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(reduced_X)
res=kmeans.fit_predict(X_scaled)
lable_pred=kmeans.labels_
centroids=kmeans.cluster_centers_
inertia=kmeans.inertia_
kmeans.fit(X_scaled)

array = [[0 for x in range(5)] for y in range(5)]

def compute(index, num):
    if playerData['Pos'][index] == 'PG':
        array[num][0] += 1
    elif playerData['Pos'][index] == 'SG':
        array[num][1] += 1
    elif playerData['Pos'][index] == 'SF':
        array[num][2] += 1
    elif playerData['Pos'][index] == 'PF':
        array[num][3] += 1
    elif playerData['Pos'][index] == 'C':
        array[num][4] += 1

for i in range(len(X_scaled)):
    if int(lable_pred[i])==0:
        plt.scatter(X_scaled[i][0],X_scaled[i][1],color='red')
    if int(lable_pred[i])==1:
        plt.scatter(X_scaled[i][0],X_scaled[i][1],color='black')
    if int(lable_pred[i])==2:
        plt.scatter(X_scaled[i][0],X_scaled[i][1],color='blue')
    if int(lable_pred[i])==3:
        plt.scatter(X_scaled[i][0],X_scaled[i][1],color='g')
    if int(lable_pred[i])==4:
        plt.scatter(X_scaled[i][0],X_scaled[i][1],color='y')
    compute(i, int(lable_pred[i]))
    
print(array)
sum = 0
for i in range(5):
    for j in range(5):
        sum += int(array[i][j])
plt.scatter(X_scaled[525][0], X_scaled[525][1], s=200, c='m', marker='*')

plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='c', marker='x')
print(sum)
plt.axis([-0.05, 1.05, -0.05, 1.05]) 
plt.show()
