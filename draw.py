import pandas as pd
import folium
from folium.plugins import MarkerCluster

# 读取数据文件
# 假设文件名为 "gowalla_data.txt"，分隔符为任意空白字符
df = pd.read_csv("Gowalla_filtered.txt", 
                 sep="\s+", 
                 header=None, 
                 names=["UserID", "Timestamp","Latitude", "Longitude", "LocationID" ])

# 按用户分组，每个用户只保留第一个数据点
df_unique = df.groupby("UserID", as_index=False).first()

# 查看处理后的数据前几行
print(df_unique.head())

# 创建一个以世界中心为起点的 folium 地图
world_map = folium.Map(location=[20, 0], zoom_start=2)

# 遍历每个用户的签到数据，并添加到地图上
for _, row in df_unique.iterrows():
    lat = row["Latitude"]
    lon = row["Longitude"]
    folium.CircleMarker(
        location=[lat, lon],
        radius=4,  # 可以根据需要调整标记大小
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.7,
        popup=f"UserID: {row['UserID']}\nLocationID: {row['LocationID']}\nTime: {row['Timestamp']}"
    ).add_to(world_map)

# 保存地图到 HTML 文件中
world_map.save("gowalla_users_map.html")
print("地图已保存到 gowalla_users_map.html")
