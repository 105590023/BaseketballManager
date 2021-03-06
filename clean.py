import json
import csv

# 載入原始檔案
with open('data.json', 'r') as rf:
    data = json.load(rf)

# print(type(data[1]["2P%"]))
# print(type(data))

# 如果同一球員在多個球隊，則將它們資料合成一個
for i in range(len(data)):
    for j in range(i+1, len(data)):
        if(data[i].get("Rk") == data[j].get("Rk") and data[i].get("Rk") != 0):
            data[i]["GS"] = float(data[i].get("GS")) + float(data[j].get("GS"))

            data[i]["MP"] = (float(data[i].get("MP"))*float(data[i].get("G")) + float(data[j].get("MP"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["FG"] = (float(data[i].get("FG"))*float(data[i].get("G")) + float(data[j].get("FG"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["FGA"] = (float(data[i].get("FGA"))*float(data[i].get("G")) + float(data[j].get("FGA"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["FG%"] = (float(data[i].get("FG%"))*float(data[i].get("G")) + float(data[j].get("FG%"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["3P"] = (float(data[i].get("3P"))*float(data[i].get("G")) + float(data[j].get("3P"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["3PA"] = (float(data[i].get("3PA"))*float(data[i].get("G")) + float(data[j].get("3PA"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["3P%"] = (float(data[i].get("3P%"))*float(data[i].get("G")) + float(data[j].get("3P%"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["2P"] = (float(data[i].get("2P"))*float(data[i].get("G")) + float(data[j].get("2P"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["2PA"] = (float(data[i].get("2PA"))*float(data[i].get("G")) + float(data[j].get("2PA"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["2P%"] = (float(data[i].get("2P%"))*float(data[i].get("G")) + float(data[j].get("2P%"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["eFG%"] = (float(data[i].get("eFG%"))*float(data[i].get("G")) + float(data[j].get("eFG%"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["FT"] = (float(data[i].get("FT"))*float(data[i].get("G")) + float(data[j].get("FT"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["FTA"] = (float(data[i].get("FTA"))*float(data[i].get("G")) + float(data[j].get("FTA"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["FT%"] = (float(data[i].get("FT%"))*float(data[i].get("G")) +float( data[j].get("FT%"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["ORB"] = (float(data[i].get("ORB"))*float(data[i].get("G")) + float(data[j].get("ORB"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["DRB"] = (float(data[i].get("DRB"))*float(data[i].get("G")) + float(data[j].get("DRB"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["TRB"] = (float(data[i].get("TRB"))*float(data[i].get("G")) + float(data[j].get("TRB"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["AST"] = (float(data[i].get("AST"))*float(data[i].get("G")) + float(data[j].get("AST"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["STL"] = (float(data[i].get("STL"))*float(data[i].get("G")) + float(data[j].get("STL"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["BLK"] = (float(data[i].get("BLK"))*float(data[i].get("G")) + float(data[j].get("BLK"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["TOV"] = (float(data[i].get("TOV"))*float(data[i].get("G")) + float(data[j].get("TOV"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["PF"] = (float(data[i].get("PF"))*float(data[i].get("G")) + float(data[j].get("PF"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["PTS"] = (float(data[i].get("PTS"))*float(data[i].get("G")) + float(data[j].get("PTS"))*float(data[j].get("G")))/(float(data[i].get("G")) + float(data[j].get("G")))

            data[i]["G"] = float(data[i].get("G")) + float(data[j].get("G"))
            data[j]["Rk"] = 0

# 刪除重複的球員
for player in list(data):
    if(player.get("Rk") == 0):
        data.remove(player)

# print(type(data[0]))
# csv欄位名稱
fieldnames = ['Rk', 'Player', 'Pos', 'Age', 'Tm',	'G',	'GS',	'MP',	'FG',	'FGA',	'FG%',	'3P',	'3PA',	'3P%',	'2P',	'2PA',	'2P%',	'eFG%',	'FT',	'FTA',	'FT%',	'ORB',	'DRB',	'TRB',	'AST',	'STL',	'BLK',	'TOV',	'PF',	'PTS']

# 開啟寫入檔
with open('csv_data.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames = fieldnames)
    writer.writeheader()
    for player in data:
        # print(type(player))
        writer.writerow(player)


# new_data = open("new_data.json", 'w')
# convert_csv = new_data.to_csv()
# csv_data = open("csv_data.csv", 'w')
# new_data.write(json.dumps(data))
# csv_data.write()
