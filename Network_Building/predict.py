import pandas as pd
import numpy as np
import math
import bisect
import random
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import folium
import pdb

# ---------------------------
# 1. 数据读取和预处理
# ---------------------------
# 读取节点数据（假设以制表符分隔）
nodes_file = 'Gowalla_filtered_nyc.txt'
nodes = pd.read_csv(nodes_file, sep='\t', header=None,
                    names=['user', 'check_in_time', 'latitude', 'longitude', 'location_id'])

# 读取好友边数据（假设以制表符分隔）
edges_file = 'Gowalla_edges_filtered_nyc.txt'
edges = pd.read_csv(edges_file, sep='\t', header=None,
                    names=['start_user', 'end_user'])

print("Nodes DataFrame:")
print(nodes.head())
print("\nEdges DataFrame:")
print(edges.head())

# 将签到时间转换为时间戳（秒）
nodes['check_in_time'] = pd.to_datetime(nodes['check_in_time'])
nodes['timestamp'] = nodes['check_in_time'].astype(np.int64) // 10**9

# ---------------------------
# 增加 cell size 处理（网格化经纬度）
# ---------------------------
cell_size = 0.001  # 单位为度， 0.001° 约对应 110m×110m
nodes['cell_x'] = np.floor(nodes['latitude'] / cell_size).astype(int)
nodes['cell_y'] = np.floor(nodes['longitude'] / cell_size).astype(int)
nodes['cell_id'] = nodes['cell_x'].astype(str) + "_" + nodes['cell_y'].astype(str)
# 如有需要，可以查看新的 cell_id 列
print("Nodes with cell_id:")
print(nodes[['latitude', 'longitude', 'cell_id']].head())

# ---------------------------
# 2. 计算每个位置（cell）的地点熵 H(l)
# ---------------------------
# 按照 cell_id 分组计算地点熵
loc_groups = nodes.groupby('cell_id')
location_entropy = {}
for cell, group in loc_groups:
    total = len(group)
    user_counts = group['user'].value_counts().to_dict()
    H = 0.0
    for count in user_counts.values():
        p = count / total
        H -= p * math.log(p + 1e-10)
    location_entropy[cell] = H

print("Sample location entropy:")
print(dict(list(location_entropy.items())[:5]))

# ---------------------------
# 3. 构建用户签到字典（基于 cell_id）
# ---------------------------
# 结构：user_checkins[user][cell_id] = sorted list of timestamps
user_checkins = {}
for row in nodes.itertuples(index=False):
    user = row.user
    # 使用网格化后的 cell_id 替换原 location_id
    loc = row.cell_id
    ts = row.timestamp
    user_checkins.setdefault(user, {}).setdefault(loc, []).append(ts)

for user in user_checkins:
    for loc in user_checkins[user]:
        user_checkins[user][loc].sort()

# ---------------------------
# 4. 辅助函数：计算两个有序时间列表的 TIS
# ---------------------------
def nearest_diff(t, sorted_list):
    pos = bisect.bisect_left(sorted_list, t)
    diffs = []
    if pos < len(sorted_list):
        diffs.append(abs(t - sorted_list[pos]))
    if pos > 0:
        diffs.append(abs(t - sorted_list[pos - 1]))
    return min(diffs) if diffs else None

def compute_tis(times1, times2):
    diffs = []
    for t in times1:
        diff = nearest_diff(t, times2)
        if diff is not None:
            diffs.append(diff)
    for t in times2:
        diff = nearest_diff(t, times1)
        if diff is not None:
            diffs.append(diff)
    if len(diffs) == 0:
        return None, None, None
    return min(diffs), max(diffs), np.mean(diffs)

# ---------------------------
# 5. 构造候选用户对及特征提取
# ---------------------------
# 平滑正负样本
# 正样本：使用 edges 文件（无向关系）
positive_pairs = set()
for row in edges.itertuples(index=False):
    positive_pairs.add(frozenset([row.start_user, row.end_user]))

# 负样本：随机采样非好友对
all_users = list(user_checkins.keys())
negative_pairs = set()
num_pos = len(positive_pairs)
while len(negative_pairs) < num_pos:
    u, v = random.sample(all_users, 2)
    pair = frozenset([u, v])
    if pair in positive_pairs:
        continue
    if set(user_checkins[u].keys()).intersection(set(user_checkins[v].keys())):
        negative_pairs.add(pair)

print("Number of positive pairs:", len(positive_pairs))
print("Number of negative pairs:", len(negative_pairs))

def compute_features(u, v):
    if u not in user_checkins or v not in user_checkins:
        return None
    common_locations = set(user_checkins[u].keys()).intersection(set(user_checkins[v].keys()))
    if len(common_locations) == 0:
        return None
    max_list = []
    min_list = []
    mean_list = []
    WL = 0.0  # weighted number of co-locations
    WO = 0.0  # weighted number of co-occurrences
    for loc in common_locations:
        times_u = user_checkins[u][loc]
        times_v = user_checkins[v][loc]
        t_min, t_max, t_mean = compute_tis(times_u, times_v)
        if t_min is None:
            continue
        max_list.append(t_max)
        min_list.append(t_min)
        mean_list.append(t_mean)
        # 使用 cell_id 对应的地点熵
        H = location_entropy.get(loc, 0)
        weight = math.exp(-H)
        WL += weight
        co_occurrence = min(len(times_u), len(times_v))
        WO += co_occurrence * weight
    if len(max_list) == 0:
        return None
    avg_max = np.mean(max_list)
    avg_min = np.mean(min_list)
    avg_mean = np.mean(mean_list)
    return [avg_max, avg_min, avg_mean, WL, WO]

features = []
labels = []
pairs_info = []
for pair in positive_pairs:
    u, v = list(pair)
    feat = compute_features(u, v)
    if feat is not None:
        features.append(feat)
        labels.append(1)
        pairs_info.append((u, v))
for pair in negative_pairs:
    u, v = list(pair)
    feat = compute_features(u, v)
    if feat is not None:
        features.append(feat)
        labels.append(0)
        pairs_info.append((u, v))

print("Total candidate pairs with common locations:", len(features))
features = np.array(features)
labels = np.array(labels)

# ---------------------------
# 6. 构建并评估逻辑回归模型
# ---------------------------
# 为了保留候选对的索引信息，构造索引数组后再进行拆分
indices = np.arange(len(features))
train_idx, test_idx, y_train, y_test = train_test_split(indices, labels, test_size=0.3, random_state=42)
X_train = features[train_idx]
X_test = features[test_idx]
pairs_info_test = [pairs_info[i] for i in test_idx]

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Test Accuracy:", score)
cv_scores = cross_val_score(clf, features, labels, cv=5)
print("Cross Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))


# ---------------------------
# 7. 自定义决策逻辑：添加额外阈值
# ---------------------------
# 对测试集进行概率预测
probs_test = clf.predict_proba(X_test)[:, 1]
custom_predictions = []

for p in probs_test:
    if p < 0.5:
        custom_predictions.append(0)        # 非好友
    elif p < 0.65:
        custom_predictions.append(1)        # 普通好友
    else:
        custom_predictions.append(2)        # 非常亲密好友

import collections
print("Custom classification distribution:", collections.Counter(custom_predictions))
from sklearn.metrics import accuracy_score
binary_custom = [1 if x > 0 else 0 for x in custom_predictions]
binary_accuracy = accuracy_score(y_test, binary_custom)
print("Binary accuracy using custom thresholds (treating 1 and 2 as positive):", binary_accuracy)

# ---------------------------
# 8. 可视化：在地图上标出节点的首次签到位置，并根据预测结果画出节点之间的连接边
# ---------------------------
# 提取每个用户的首次签到位置
first_checkins = nodes.sort_values('check_in_time').groupby('user').first().reset_index()
# 构造字典：user -> (latitude, longitude)
user_locations = dict(zip(first_checkins['user'], zip(first_checkins['latitude'], first_checkins['longitude'])))

# 创建地图（这里以纽约市中心为例）
map_center = [40.71, -74.0]  # 纽约市中心
m = folium.Map(location=map_center, zoom_start=12)

# 将每个用户的首次签到位置标记在地图上
for idx, row in first_checkins.iterrows():
    user = row['user']
    lat = row['latitude']
    lon = row['longitude']
    folium.CircleMarker(location=[lat, lon],
                        radius=3,
                        color='yellow',
                        fill=True,
                        fill_color='yellow',
                        popup=f"User: {user}").add_to(m)

# 根据测试集中的候选对及预测结果，连接好友
# 其中：1（普通好友）用蓝色边，2（非常亲密好友）用红色边
for pred, pair in zip(custom_predictions, pairs_info_test):
    if pred > 0:  # 仅连接预测为好友的
        u, v = list(pair)
        if u in user_locations and v in user_locations:
            latlon_u = user_locations[u]
            latlon_v = user_locations[v]
            color = 'blue' if pred == 1 else 'red'
            folium.PolyLine(locations=[latlon_u, latlon_v], color=color, weight=2, opacity=0.8).add_to(m)

# 保存地图
m.save("predicted_friendship_map.html")
print("Map saved as predicted_friendship_map.html")

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay

# 1. 使用训练好的模型对测试集进行预测概率
#    注意：Precision-Recall曲线一般基于预测概率，而不是二分类的硬预测
y_scores = clf.predict_proba(X_test)[:, 1]  # 取正类的概率

# 2. 计算Precision、Recall以及阈值
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# 3. 计算平均精确率AP（Average Precision）
ap = average_precision_score(y_test, y_scores)

# 4. 使用 PrecisionRecallDisplay 绘制PR曲线
disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=ap)
disp.plot()
plt.title("Precision-Recall (PR) Curve")
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 使用训练好的模型对测试集进行预测
y_pred = clf.predict(X_test)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# 使用 ConfusionMatrixDisplay 绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import precision_recall_fscore_support, classification_report

# 使用模型对测试集进行预测
y_pred = clf.predict(X_test)

# 计算每个类别的 precision, recall, f1, 和 support
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

print("Class 0 Metrics:")
print("Precision: {:.4f}".format(precision[0]))
print("Recall:    {:.4f}".format(recall[0]))
print("F1 Score:  {:.4f}".format(f1[0]))
print("Support:   {}".format(support[0]))

print("\nClass 1 Metrics:")
print("Precision: {:.4f}".format(precision[1]))
print("Recall:    {:.4f}".format(recall[1]))
print("F1 Score:  {:.4f}".format(f1[1]))
print("Support:   {}".format(support[1]))

# 或者打印详细的分类报告
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
