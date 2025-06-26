# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.dates as mdates
# from matplotlib.ticker import FuncFormatter
# import pdb

# df_geo = pd.read_csv('single_geocode_result.csv')
# df_main = pd.read_csv('data/RecordList202309.csv', encoding='latin1')

# # 先统一地址格式（去除大小写、空格影响）
# df_geo['address_clean'] = df_geo['address'].str.upper().str.strip()
# df_main['address_clean'] = df_main['Address'].str.upper().str.strip()
# df_geo = df_geo.drop_duplicates(subset='address_clean')
# # 用 address_clean 字段合并
# df_merged = pd.merge(df_main, df_geo[['address_clean', 'lon', 'lat']], on='address_clean', how='left')

# # 可选：删除中间列
# df_merged.drop(columns='address_clean', inplace=True)

# # 保存结果

# # 提取年份（例如从 ROF2022-00001 提取出 2022）
# df_merged['year'] = df_merged['Record Number'].str.extract(r'(\d{4})')
# # 只保留有效年份（比如大于1900）
# df_merged = df_merged[df_merged['year'].notna() & (df_merged['year'].astype(int) >= 1900)]

# # 转换为日期，默认使用 1 月 1 日
# df_merged['record_date'] = pd.to_datetime(df_merged['year'] + '-09-01')

# # 如果你只想要 'YYYY-MM-DD' 格式的字符串：
# df_merged['record_date'] = df_merged['record_date'].dt.strftime('%Y-%m-%d')


# df_merged.drop(columns='year', inplace=True)
# df_merged.drop(columns='Record Number', inplace=True)
# df_merged.drop(columns='Address', inplace=True)

# print(df_merged.head())
# df_merged.to_csv("merged_with_coordinates_202309.csv", index=False)



import pandas as pd
import glob

# 找到所有匹配的 CSV 文件（根据你命名格式）
csv_files = sorted(glob.glob("data/merged_with_coordinates_202*.csv"))

# 读取并合并所有文件
df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# 保存合并后的结果
df_all.to_csv("data/merged_with_coordinates.csv", index=False)

print(f"✅ 合并完成，共 {len(df_all)} 条记录，保存为 merged_with_coordinates.csv")