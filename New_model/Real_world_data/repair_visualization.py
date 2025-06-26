import pandas as pd
import matplotlib.pyplot as plt
import folium
import numpy as np
from datetime import datetime
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import os

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



def create_repair_animation():
    """
    创建从2020年8月到2021年9月的repair数据动画
    """
    # 生成时间范围
    time_range = pd.date_range(start="2022-07-01", end="2023-05-31", freq="MS")  # 月初为频率
    
    # 创建保存路径
    gif_path = "repair_animation.gif"
    
    # 创建图像和子图
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    
    def update(month):
        ax.clear()
        # 格式化当前月份
        year = month.year
        month_num = month.month
        
        # 构建文件名
        if month_num < 10:
            month_str = f"0{month_num}"
        else:
            month_str = str(month_num)
        
        data_file = f"data/merged_with_coordinates_{year}{month_str}.csv"
        
        # 检查文件是否存在
        if not os.path.exists(data_file):
            # 如果文件不存在，显示空白图
            ax.set_title(f"No data available for {month.strftime('%B %Y')}", 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
            ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#BDC3C7')
            ax.spines['bottom'].set_color('#BDC3C7')
            ax.set_facecolor('#F8F9FA')
            return
        
        try:
            # 读取数据
            df = pd.read_csv(data_file)
            
            # 过滤掉没有经纬度的记录
            df = df.dropna(subset=['lon', 'lat'])
            
            # 确保经纬度是数值类型
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df = df.dropna(subset=['lon', 'lat'])
            
            # 绘制散点图
            ax.scatter(df['lon'], df['lat'], 
                      c='black', s=15, alpha=0.7, label='Houses with Repair Records')
            
            # 设置标题和标签
            ax.set_title(f'House Repair Distribution - {month.strftime("%B %Y")}', 
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
            total_records = len(pd.read_csv(data_file))
            
            info_text = f'Total Records: {total_records}\nWith Coordinates: {total_houses}\nCoverage Rate: {total_houses/total_records*100:.1f}%'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#BDC3C7'))
            
        except Exception as e:
            # 如果读取数据出错，显示错误信息
            ax.set_title(f'Error loading data for {month.strftime("%B %Y")}', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
            ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#BDC3C7')
            ax.spines['bottom'].set_color('#BDC3C7')
            ax.set_facecolor('#F8F9FA')
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12, color='red')
    
    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=time_range, interval=2000, repeat=True)
    
    # 保存 GIF
    writer = PillowWriter(fps=0.5)  # 每帧2秒
    ani.save(gif_path, writer=writer)
    print(f"动画已保存到: {gif_path}")
    
    # 显示动画
    plt.show()


create_repair_animation()
# ----------- PARAMETERS -----------
csv_path = 'data/merged_with_coordinates.csv'       # make sure this file path is correct
gif_path = 'repair_heatmap.gif'      # output gif
grid_size = 0.005                              # 0.005° ≈ 500 m grid
start_date, end_date = "2022-07-01", "2023-05-31"

# polygon rectangle (Lee County bbox you provided)
min_lon, max_lon = -82.04150698533131, -81.87502336164845
min_lat, max_lat =  26.490448860026532, 26.604200914701607

# ----------- 1. LOAD & FILTER -----------
df = pd.read_csv(csv_path)
df = df.dropna(subset=['lon', 'lat', 'record_date']).copy()

# bbox filter
df = df[(df['lon'].between(min_lon, max_lon)) & (df['lat'].between(min_lat, max_lat))]

# date parse/filter
df['record_date'] = pd.to_datetime(df['record_date'], errors='coerce')
df = df[df['record_date'].between(start_date, end_date)]

# ----------- 2. GRID & TIME BINS -----------
lon_bins = np.arange(min_lon, max_lon + grid_size, grid_size)
lat_bins = np.arange(min_lat, max_lat + grid_size, grid_size)

df['lon_bin'] = pd.cut(df['lon'], bins=lon_bins, labels=False)
df['lat_bin'] = pd.cut(df['lat'], bins=lat_bins, labels=False)

time_range = pd.date_range(start=start_date, end=end_date, freq='MS')

monthly_counts = []
max_count = 0
for month in time_range:
    month_data = df[(df['record_date'].dt.year == month.year) &
                    (df['record_date'].dt.month == month.month)]
    grouped = month_data.groupby(['lon_bin', 'lat_bin']).size().unstack(fill_value=0)
    monthly_counts.append(grouped)
    if not grouped.empty:
        max_count = max(max_count, grouped.values.max())

# ----------- 3. INITIAL PLOT -----------
fig, ax = plt.subplots(figsize=(8, 7))
initial_heat = np.zeros((len(lon_bins)-1, len(lat_bins)-1))
im = ax.imshow(initial_heat.T, origin='lower', cmap='hot', interpolation='nearest',
               extent=[min_lon, max_lon, min_lat, max_lat],
               vmin=0, vmax=max_count)
cb = plt.colorbar(im, ax=ax, label='Repairs per grid')

ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title('Repair Heatmap')

def update(frame_idx):
    heat = monthly_counts[frame_idx]
    full_grid = np.zeros((len(lon_bins)-1, len(lat_bins)-1))
    for x in heat.index:
        for y in heat.columns:
            full_grid[int(x), int(y)] = heat.loc[x, y]
    im.set_data(full_grid.T)
    ax.set_title(f"House Repair Heatmap - {time_range[frame_idx].strftime('%B %Y')}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(time_range), interval=1000, blit=False)
ani.save(gif_path, writer=PillowWriter(fps=0.75))
plt.close(fig)