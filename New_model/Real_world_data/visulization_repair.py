import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib import animation
from matplotlib.animation import PillowWriter
import pdb
import folium
from datetime import datetime

def combined_repair_plots(
        csv_path='data/merged_with_coordinates.csv',
        start='2022-07-01',
        end='2023-09-30',
        grid_size=0.01,
        save_path='combined_repair_plots.png'):
    """
    读取 merged_with_coordinates.csv，绘制：
      ① 每月修复数量折线
      ② 同网格内修复数量 Gini 系数折线
    仅分析落在指定 polygon 内、且 lon/lat 非空的记录
    """

    # ---------- ① 读入并先剔除空 lon/lat ----------
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['lon', 'lat', 'record_date']).copy()

    # ---------- ② 坐标范围过滤 ----------
    # Lee County 矩形 polygon（经纬度边界）
    min_lon, max_lon = -82.04150698533131, -81.87502336164845
    min_lat, max_lat =  26.490448860026532, 26.604200914701607
    df = df[(df['lon'].between(min_lon, max_lon)) &
            (df['lat'].between(min_lat, max_lat))]

    # ---------- ③ 日期预处理 ----------
    df['record_date'] = pd.to_datetime(df['record_date'], errors='coerce')
    df = df.dropna(subset=['record_date'])
    df = df[df['record_date'].between(start, end)]

    # ---------- ④ 以下保持与之前一致 ----------
    # 每月修复数量
    monthly_cnt = (
        df.groupby(df['record_date'].dt.to_period('M'))
          .size()
          .rename('repairs')
    )
    monthly_cnt.index = monthly_cnt.index.to_timestamp()

    # Gini 计算函数
    def gini(arr):
        if len(arr) == 0 or arr.sum() == 0: return 0
        arr = np.sort(arr); n = arr.size
        return (2*(np.arange(1,n+1)*arr).sum())/(n*arr.sum()) - (n+1)/n

    # 网格划分
    lon_bins = np.arange(df['lon'].min(), df['lon'].max()+grid_size, grid_size)
    lat_bins = np.arange(df['lat'].min(), df['lat'].max()+grid_size, grid_size)
    df['block_id'] = (pd.cut(df['lon'], lon_bins, labels=False).astype(str) + '_' +
                      pd.cut(df['lat'], lat_bins, labels=False).astype(str))

    # 月度 Gini
    rng = pd.date_range(start=start, end=end, freq='MS')
    gini_vals, month_lbls = [], []
    for m in rng:
        sub = df[(df['record_date'].dt.year==m.year)&(df['record_date'].dt.month==m.month)]
        gini_vals.append(gini(sub['block_id'].value_counts().values))
        month_lbls.append(m.strftime('%Y-%m'))

    # ---------- 绘图 ----------
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(18,10),facecolor='white')

    # 修复数量
    ax1.plot(monthly_cnt.index, monthly_cnt.values,'o-',
             lw=3,ms=8,color='#2E86AB',mfc='#A23B72',mec='white',mew=2)
    for d,v in zip(monthly_cnt.index, monthly_cnt.values):
        ax1.annotate(v,xy=(d,v),xytext=(0,10),textcoords='offset points',
                     ha='center',fontsize=10,color='#2E86AB')
    ax1.set_title('Monthly House Repair Count (Filtered Area)',fontsize=18,pad=20)
    ax1.set_ylabel('Repairs',fontsize=14)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{int(x):,}'))
    ax1.grid(True,ls='--',alpha=.3); ax1.set_facecolor('#F8F9FA')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(),rotation=45,ha='right')

    # Gini
    ax2.plot(month_lbls,gini_vals,'s-',lw=2,ms=8,color='#E74C3C',label='Gini')
    ax2.set_title('Monthly Repair Inequality Across Blocks (Gini)',fontsize=18,pad=20)
    ax2.set_xlabel('Month',fontsize=14); ax2.set_ylabel('Gini Coefficient',fontsize=14)
    ax2.grid(True,ls='--',alpha=.3); ax2.set_facecolor('#F8F9FA')
    plt.setp(ax2.xaxis.get_majorticklabels(),rotation=45,ha='right')
    ax2.legend(fontsize=12)

    plt.tight_layout(); plt.savefig(save_path,dpi=300,bbox_inches='tight')
    plt.show()

    # 摘要
    print(f"Records kept: {len(df):,}")
    print(f"Monthly repairs (sum): {monthly_cnt.sum():,}")
    print(f"Gini avg/min/max: {np.mean(gini_vals):.3f} / {np.min(gini_vals):.3f} / {np.max(gini_vals):.3f}")

# 调用示例
# combined_repair_plots()


csv_path = 'data/merged_with_coordinates.csv'

# ---------- 1. 读取并过滤 ----------
df = pd.read_csv(csv_path)
min_lon, max_lon = -82.04150698533131, -81.87502336164845
min_lat, max_lat =  26.490448860026532, 26.604200914701607
df = df[(df['lon'].between(min_lon, max_lon)) &
        (df['lat'].between(min_lat, max_lat))]


def create_repair_map(data_file="data/merged_with_coordinates_202008.csv", 
                     output_file="august_repair_map.html",
                     target_year=2020, 
                     target_month=8,
                     radius=2,
                     zoom_start=11):
    """
    创建8月份repair数据地图可视化
    
    参数:
    data_file (str): 数据文件路径
    output_file (str): 输出HTML文件路径
    target_year (int): 目标年份
    target_month (int): 目标月份
    radius (int): 点的大小
    zoom_start (int): 地图缩放级别
    """
    
    # 读取数据
    df = pd.read_csv(data_file)
    
    # 过滤掉没有经纬度的记录
    df = df.dropna(subset=['lon', 'lat'])
    
    # 确保经纬度是数值类型
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df = df.dropna(subset=['lon', 'lat'])
    
    # 判断是否为指定年月
    def is_target_period(date_str):
        if pd.isna(date_str) or date_str == '':
            return False
        try:
            date = pd.to_datetime(date_str)
            return date.year == target_year and date.month == target_month
        except:
            return False
    
    # 应用判断并添加颜色列
    df['color'] = df['record_date'].apply(
        lambda x: 'black' if is_target_period(x) else 'gray'
    )
    
    # 创建 folium 地图
    center_lat = float(df['lat'].mean())
    center_lon = float(df['lon'].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    
    # 添加每个点
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=(float(row['lat']), float(row['lon'])),
            radius=radius,
            color=str(row['color']),
            fill=True,
            fill_color=str(row['color']),
            fill_opacity=0.8,
            weight=0
        ).add_to(m)
    
    # 保存为 HTML 文件
    m.save(output_file)
    print(f"地图已保存到: {output_file}")
    
    # 打印统计信息
    total_houses = len(df)
    houses_with_repair = len(df[df['color'] == 'black'])
    houses_without_repair = len(df[df['color'] == 'gray'])
    
    print(f"\n=== August 2020 Repair Statistics ===")
    print(f"Total Houses: {total_houses}")
    print(f"Houses with Repair Records: {houses_with_repair}")
    print(f"Houses without Repair Records: {houses_without_repair}")
    print(f"Repair Rate: {houses_with_repair/total_houses*100:.2f}%")

def create_repair_scatter_plot(data_file="data/merged_with_coordinates_202008.csv",
                              output_file="august_repair_scatter.png",
                              target_year=2020,
                              target_month=8):
    """
    创建8月份repair数据的散点图
    
    参数:
    data_file (str): 数据文件路径
    output_file (str): 输出PNG文件路径
    target_year (int): 目标年份
    target_month (int): 目标月份
    """
    
    # 读取数据
    df = pd.read_csv(data_file)
    
    # 过滤掉没有经纬度的记录
    df = df.dropna(subset=['lon', 'lat'])
    
    # 确保经纬度是数值类型
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df = df.dropna(subset=['lon', 'lat'])
    
    # 判断是否为指定年月
    def is_target_period(date_str):
        if pd.isna(date_str) or date_str == '':
            return False
        try:
            date = pd.to_datetime(date_str)
            return date.year == target_year and date.month == target_month
        except:
            return False
    
    # 应用判断并添加颜色列
    df['color'] = df['record_date'].apply(
        lambda x: 'black' if is_target_period(x) else 'gray'
    )
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    
    # 绘制散点图
    for color in ['gray', 'black']:
        subset = df[df['color'] == color]
        label = 'With Repair Record' if color == 'black' else 'Without Repair Record'
        ax.scatter(subset['lon'], subset['lat'], 
                  c=color, s=20, alpha=0.7, label=label)
    
    # 设置标题和标签
    ax.set_title(f'August {target_year} House Repair Distribution', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    
    # 添加图例
    ax.legend(fontsize=10)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    # 添加背景色
    ax.set_facecolor('#F8F9FA')
    
    # 添加统计信息
    total_houses = len(df)
    houses_with_repair = len(df[df['color'] == 'black'])
    houses_without_repair = len(df[df['color'] == 'gray'])
    
    info_text = f'Total Houses: {total_houses}\nWith Repair: {houses_with_repair}\nWithout Repair: {houses_without_repair}\nRepair Rate: {houses_with_repair/total_houses*100:.1f}%'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#BDC3C7'))
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 打印统计信息
    print(f"\n=== August {target_year} Repair Statistics ===")
    print(f"Total Houses: {total_houses}")
    print(f"Houses with Repair Records: {houses_with_repair}")
    print(f"Houses without Repair Records: {houses_without_repair}")
    print(f"Repair Rate: {houses_with_repair/total_houses*100:.2f}%")


# 
combined_repair_plots()