import pandas as pd
import matplotlib.pyplot as plt
import folium
import matplotlib.animation as animation
from datetime import datetime
import os
import numpy as np
from matplotlib.animation import PillowWriter

def create_sales_map(data_file="data/sales_data_with_first_recent_sale.csv", 
                    output_file="august_sales_map.html",
                    target_year=2020, 
                    target_month=8,
                    radius=2,
                    zoom_start=11):
    """
    创建销售数据地图可视化
    
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
    
    # 判断是否为指定年月
    def is_target_period(date_str):
        if date_str == '0':
            return False
        try:
            date = pd.to_datetime(date_str)
            return date.year == target_year and date.month == target_month
        except:
            return False
    
    # 应用判断并添加颜色列
    df['color'] = df['first_sale_in_period'].apply(
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


def sales_animation():
    # 加载数据
    df = pd.read_csv('data/sales_data_with_first_recent_sale.csv')

    # 生成时间范围
    time_range = pd.date_range(start="2022-07-01", end="2023-09-30", freq="MS")  # 月初为频率

    # 预处理日期列
    df['first_sale_in_period'] = df['first_sale_in_period'].astype(str)

    # 创建保存路径
    gif_path = "sales_animation.gif"

    # 创建图像和子图
    fig, ax = plt.subplots(figsize=(10, 8))

    def update(month):
        ax.clear()
        # 格式化当前月份
        year = month.year
        month_num = month.month

        # 设置颜色
        def is_in_month(date_str):
            if date_str == '0':
                return False
            try:
                date = pd.to_datetime(date_str)
                return date.year == year and date.month == month_num
            except:
                return False

        df['color'] = df['first_sale_in_period'].apply(
            lambda x: 'black' if is_in_month(x) else 'gray'
        )

        ax.scatter(df['lon'], df['lat'], c=df['color'], s=5)
        ax.set_title(f"Sales in {month.strftime('%B %Y')} (Black=Sold, Gray=No Sale)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True)

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=time_range, interval=2000, repeat=False)  # 2秒一帧

    writer = PillowWriter(fps=0.5)  # 每帧2秒
    ani.save("sales_animation.gif", writer=writer)

def sales_heatmap():
    # 读取数据
    df = pd.read_csv('data/sales_data_with_first_recent_sale.csv')
    df['first_sale_in_period'] = df['first_sale_in_period'].astype(str)

    # 过滤有效销售记录并转时间
    valid_sales = df[df['first_sale_in_period'] != '0'].copy()
    valid_sales['sale_date'] = pd.to_datetime(valid_sales['first_sale_in_period'], errors='coerce')

    # 定义网格大小（经度和纬度方向各0.01度）
    lon_bins = np.arange(df['lon'].min(), df['lon'].max() + 0.005, 0.005)
    lat_bins = np.arange(df['lat'].min(), df['lat'].max() + 0.005, 0.005)

    # 给每条记录打上grid编号
    valid_sales['lon_bin'] = pd.cut(valid_sales['lon'], bins=lon_bins, labels=False)
    valid_sales['lat_bin'] = pd.cut(valid_sales['lat'], bins=lat_bins, labels=False)

    # 创建时间序列
    time_range = pd.date_range(start="2022-07-01", end="2023-09-30", freq="MS")

    # 预先计算每个月每个网格的销售量
    max_count = 0
    monthly_counts = []

    for month in time_range:
        month_data = valid_sales[
            (valid_sales['sale_date'].dt.year == month.year) &
            (valid_sales['sale_date'].dt.month == month.month)
        ]
        grouped = month_data.groupby(['lon_bin', 'lat_bin']).size().unstack(fill_value=0)
        monthly_counts.append(grouped)
        if not grouped.empty:
            max_count = max(max_count, grouped.values.max())

    # 初始化图像和热图
    fig, ax = plt.subplots(figsize=(10, 8))
    initial_heat = np.zeros((len(lon_bins)-1, len(lat_bins)-1))
    heatmap = ax.imshow(
        initial_heat.T,
        origin='lower',
        cmap='hot',
        interpolation='nearest',
        extent=[lon_bins.min(), lon_bins.max(), lat_bins.min(), lat_bins.max()],
        vmin=0,
        vmax=max_count
    )
    colorbar = plt.colorbar(heatmap, ax=ax, label='Number of Sales')

    # update 函数用于更新每一帧
    def update(i):
        heat = monthly_counts[i]
        # 创建完整的空网格，填入已有数据
        full_grid = np.zeros((len(lon_bins)-1, len(lat_bins)-1))
        for x in heat.index:
            for y in heat.columns:
                full_grid[x, y] = heat.loc[x, y]
        heatmap.set_data(full_grid.T)
        ax.set_title(f"Sales Heatmap - {time_range[i].strftime('%B %Y')}")
        return [heatmap]

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(time_range), interval=2000, blit=False)

    # 保存 GIF —— 设定帧率 0.5 fps ⇒ 每帧 2 s
    writer = animation.PillowWriter(fps=0.75)      # PillowWriter 允许小数帧率
    ani.save("grid_sales_heatmap.gif", writer=writer)

def sales_line_plot():
    """
    Plot monthly sales count line chart
    """
    # Load data
    df = pd.read_csv('data/sales_data_with_first_recent_sale.csv')
    
    # Preprocess time column
    df['first_sale_in_period'] = df['first_sale_in_period'].astype(str)
    
    # Filter out invalid sales records (marked as '0')
    valid_sales = df[df['first_sale_in_period'] != '0'].copy()
    
    # Convert to datetime type
    valid_sales['sale_date'] = pd.to_datetime(valid_sales['first_sale_in_period'], errors='coerce')
    
    # Drop rows with invalid dates
    valid_sales = valid_sales.dropna(subset=['sale_date'])
    
    # Group by month and count sales
    monthly_sales = valid_sales.groupby(valid_sales['sale_date'].dt.to_period('M')).size()
    monthly_sales.index = monthly_sales.index.to_timestamp()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    # Plot line chart
    line = ax.plot(monthly_sales.index, monthly_sales.values, 
                   marker='o', linewidth=3, markersize=8, 
                   color='#2E86AB', markerfacecolor='#A23B72', 
                   markeredgecolor='white', markeredgewidth=2)
    
    # Add data labels
    for i, (date, value) in enumerate(zip(monthly_sales.index, monthly_sales.values)):
        ax.annotate(f'{value}', 
                   xy=(date, value), 
                   xytext=(0, 10), 
                   textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold',
                   color='#2E86AB')
    
    # Set title and labels
    ax.set_title('Monthly Sales Count', fontsize=20, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('Month', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Number of Sales', fontsize=14, fontweight='bold', color='#2C3E50')
    
    # Set grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Beautify axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    # Set x-axis date format
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Set y-axis format
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add background color
    ax.set_facecolor('#F8F9FA')
    
    # Add statistics info
    total_sales = monthly_sales.sum()
    avg_sales = monthly_sales.mean()
    max_sales = monthly_sales.max()
    max_month = monthly_sales.idxmax().strftime('%Y-%m')
    
    info_text = f'Total Sales: {total_sales:,}\nAvg Monthly: {avg_sales:.1f}\nPeak Month: {max_sales} ({max_month})'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#BDC3C7'))
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig("sales_line_plot.png", dpi=300, bbox_inches='tight', facecolor='white')
    # Show plot
    plt.show()
    
    # Print statistics
    print(f"\n=== Sales Statistics ===")
    print(f"Total Sales: {total_sales:,}")
    print(f"Average Monthly Sales: {avg_sales:.1f}")
    print(f"Peak Month: {max_sales} ({max_month})")
    print(f"Data Range: {monthly_sales.index.min().strftime('%Y-%m')} to {monthly_sales.index.max().strftime('%Y-%m')}")

def gini_coefficient_plot():
    # Gini 系数计算函数
    def gini_coefficient(x):
        if len(x) == 0:
            return 0
        x = np.sort(np.array(x))  # 排序
        n = len(x)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n if np.sum(x) != 0 else 0

    # 读取数据
    df = pd.read_csv('data/sales_data_with_first_recent_sale.csv')
    df['first_sale_in_period'] = df['first_sale_in_period'].astype(str)

    # 过滤有效销售记录并转时间
    valid_sales = df[df['first_sale_in_period'] != '0'].copy()
    valid_sales['sale_date'] = pd.to_datetime(valid_sales['first_sale_in_period'], errors='coerce')

    # 更细的网格划分（0.005 度）
    lon_bins = np.arange(df['lon'].min(), df['lon'].max() + 0.005, 0.005)
    lat_bins = np.arange(df['lat'].min(), df['lat'].max() + 0.005, 0.005)
    valid_sales['lon_bin'] = pd.cut(valid_sales['lon'], bins=lon_bins, labels=False)
    valid_sales['lat_bin'] = pd.cut(valid_sales['lat'], bins=lat_bins, labels=False)

    # 创建 block ID
    valid_sales['block_id'] = valid_sales['lon_bin'].astype(str) + "_" + valid_sales['lat_bin'].astype(str)

    # 时间序列
    time_range = pd.date_range(start="2022-07-01", end="2023-08-31", freq="MS")

    # 计算每月的 Gini 系数
    month_labels = []
    gini_values = []

    for month in time_range:
        month_data = valid_sales[
            (valid_sales['sale_date'].dt.year == month.year) &
            (valid_sales['sale_date'].dt.month == month.month)
        ]
        block_counts = month_data['block_id'].value_counts().values
        gini_values.append(gini_coefficient(block_counts))
        month_labels.append(month.strftime('%Y-%m'))

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(month_labels, gini_values, label='Gini Coefficient', marker='s', color='#2E86AB', linewidth=2, markersize=8)
    plt.xticks(rotation=45)
    plt.title("Monthly Sales Inequality Across Blocks (Gini Coefficient)", fontsize=16, fontweight='bold')
    plt.xlabel("Month", fontsize=12, fontweight='bold')
    plt.ylabel("Gini Coefficient", fontsize=12, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("gini_coefficient_plot.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def combined_sales_plots():
    """
    Create a combined plot with sales line chart on top and gini coefficient plot on bottom
    """
    # Load data
    df = pd.read_csv('data/sales_data_with_first_recent_sale.csv')
    
    # Preprocess time column
    df['first_sale_in_period'] = df['first_sale_in_period'].astype(str)
    
    # Filter out invalid sales records (marked as '0')
    valid_sales = df[df['first_sale_in_period'] != '0'].copy()
    
    # Convert to datetime type
    valid_sales['sale_date'] = pd.to_datetime(valid_sales['first_sale_in_period'], errors='coerce')
    
    # Drop rows with invalid dates
    valid_sales = valid_sales.dropna(subset=['sale_date'])
    
    # Group by month and count sales
    monthly_sales = valid_sales.groupby(valid_sales['sale_date'].dt.to_period('M')).size()
    monthly_sales.index = monthly_sales.index.to_timestamp()
    
    # Gini coefficient calculation function
    def gini_coefficient(x):
        if len(x) == 0:
            return 0
        x = np.sort(np.array(x))  # 排序
        n = len(x)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n if np.sum(x) != 0 else 0

    # Grid division for gini calculation
    lon_bins = np.arange(df['lon'].min(), df['lon'].max() + 0.005, 0.005)
    lat_bins = np.arange(df['lat'].min(), df['lat'].max() + 0.005, 0.005)
    valid_sales['lon_bin'] = pd.cut(valid_sales['lon'], bins=lon_bins, labels=False)
    valid_sales['lat_bin'] = pd.cut(valid_sales['lat'], bins=lat_bins, labels=False)
    valid_sales['block_id'] = valid_sales['lon_bin'].astype(str) + "_" + valid_sales['lat_bin'].astype(str)

    # Time range
    time_range = pd.date_range(start="2022-07-01", end="2023-09-30", freq="MS")

    # Calculate monthly Gini coefficients
    month_labels = []
    gini_values = []

    for month in time_range:
        month_data = valid_sales[
            (valid_sales['sale_date'].dt.year == month.year) &
            (valid_sales['sale_date'].dt.month == month.month)
        ]
        block_counts = month_data['block_id'].value_counts().values
        gini_values.append(gini_coefficient(block_counts))
        month_labels.append(month.strftime('%Y-%m'))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), facecolor='white')
    
    # Top subplot: Sales line chart
    line = ax1.plot(monthly_sales.index, monthly_sales.values, 
                   marker='o', linewidth=3, markersize=8, 
                   color='#2E86AB', markerfacecolor='#A23B72', 
                   markeredgecolor='white', markeredgewidth=2)
    
    # Add data labels for sales
    for i, (date, value) in enumerate(zip(monthly_sales.index, monthly_sales.values)):
        ax1.annotate(f'{value}', 
                   xy=(date, value), 
                   xytext=(0, 10), 
                   textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold',
                   color='#2E86AB')
    
    # Set title and labels for sales plot
    ax1.set_title('Monthly Sales Count', fontsize=18, fontweight='bold', pad=20, color='#2C3E50')
    ax1.set_ylabel('Number of Sales', fontsize=14, fontweight='bold', color='#2C3E50')
    
    # Set grid for sales plot
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Beautify axes for sales plot
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#BDC3C7')
    ax1.spines['bottom'].set_color('#BDC3C7')
    
    # Set y-axis format for sales plot
    from matplotlib.ticker import FuncFormatter
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add background color for sales plot
    ax1.set_facecolor('#F8F9FA')
    
    # Add statistics info for sales
    total_sales = monthly_sales.sum()
    avg_sales = monthly_sales.mean()
    max_sales = monthly_sales.max()
    max_month = monthly_sales.idxmax().strftime('%Y-%m')
    
    info_text = f'Total Sales: {total_sales:,}\nAvg Monthly: {avg_sales:.1f}\nPeak Month: {max_sales} ({max_month})'
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#BDC3C7'))
    
    # Bottom subplot: Gini coefficient plot
    ax2.plot(month_labels, gini_values, label='Gini Coefficient', marker='s', 
             color='#E74C3C', linewidth=2, markersize=8)
    
    # Set title and labels for gini plot
    ax2.set_title('Monthly Sales Inequality Across Blocks (Gini Coefficient)', 
                  fontsize=18, fontweight='bold', pad=20, color='#2C3E50')
    ax2.set_xlabel('Month', fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.set_ylabel('Gini Coefficient', fontsize=14, fontweight='bold', color='#2C3E50')
    
    # Set grid for gini plot
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # Beautify axes for gini plot
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#BDC3C7')
    ax2.spines['bottom'].set_color('#BDC3C7')
    
    # Add background color for gini plot
    ax2.set_facecolor('#F8F9FA')
    
    # Rotate x-axis labels for gini plot
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add legend for gini plot
    ax2.legend(fontsize=12)
    
    # Set x-axis date format for sales plot
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig("combined_sales_plots.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print statistics
    print(f"\n=== Sales Statistics ===")
    print(f"Total Sales: {total_sales:,}")
    print(f"Average Monthly Sales: {avg_sales:.1f}")
    print(f"Peak Month: {max_sales} ({max_month})")
    print(f"Data Range: {monthly_sales.index.min().strftime('%Y-%m')} to {monthly_sales.index.max().strftime('%Y-%m')}")
    
    print(f"\n=== Gini Coefficient Statistics ===")
    print(f"Average Gini: {np.mean(gini_values):.3f}")
    print(f"Max Gini: {np.max(gini_values):.3f}")
    print(f"Min Gini: {np.min(gini_values):.3f}")


sales_animation()
