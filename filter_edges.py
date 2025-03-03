import pandas as pd

# 1. 读取过滤后的签到数据文件，并提取 UserID 列
filtered_df = pd.read_csv(
    "Gowalla_filtered_nyc.txt",
    sep="\t",        # 若原文件是以制表符分隔，可改为 sep=" " 或根据实际情况调整
    header=None,
    names=["UserID", "Latitude", "Longitude", "LocationID", "Timestamp"]
)
# 提取唯一的 UserID 集合
unique_users = set(filtered_df["UserID"].unique())
print(f"共有 {len(unique_users)} 个用户在过滤区域内")

# 2. 读取边信息文件，假设文件名为 "Gowalla_edges.txt"，格式为：start_user end_user
edges_df = pd.read_csv(
    "Gowalla_edges.txt",
    sep="\s+",       # 以任意空白字符分隔
    header=None,
    names=["start_user", "end_user"]
)
print(f"原始边信息共有 {len(edges_df)} 条")

# 3. 筛选边信息：保留边的两端用户都在 unique_users 内的边
filtered_edges_df = edges_df[
    edges_df["start_user"].isin(unique_users) & edges_df["end_user"].isin(unique_users)
]
print(f"过滤后边信息共有 {len(filtered_edges_df)} 条")

# 4. 将过滤后的边信息保存到新的文件
filtered_edges_df.to_csv(
    "Gowalla_edges_filtered_nyc.txt",
    sep="\t",    # 以制表符分隔，可根据需要修改
    index=False,
    header=False
)

print("边信息过滤完成，结果已保存到 Gowalla_edges_filtered.txt")
