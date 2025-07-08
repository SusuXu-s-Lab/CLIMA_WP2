import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import pdb

bbox = (-81.97581623909956, 26.504952873345545, -81.95459566656734, 26.52311972089953)   # (minx, miny, maxx, maxy)

sales_df = pd.read_csv('fl/fl/sales_data_with_recent_sale.csv')
minx, miny, maxx, maxy = bbox
sales_df_filtered = sales_df[
    (sales_df['lon'] >= minx) & (sales_df['lon'] <= maxx) &
    (sales_df['lat'] >= miny) & (sales_df['lat'] <= maxy)
]

pop_subset = pd.read_csv('fl/fl/pop_subset_small.csv')

def match_households_to_sales_locations(pop_subset, sales_df_filtered):
    """
    确保每个销售点都分配给一个家庭，优先一一对应
    销售点数量：1516个，家庭数量：1761个
    目标：1516个销售点都有对应的家庭
    """
    # 获取唯一的家庭和它们的坐标
    household_coords = pop_subset.groupby('hhold')[['long', 'lat']].first().reset_index()
    print(f"唯一家庭数量: {len(household_coords)}")
    
    # 获取销售点坐标
    sales_coords = sales_df_filtered[['lon', 'lat']].drop_duplicates().reset_index(drop=True)
    print(f"唯一销售点数量: {len(sales_coords)}")
    
    # 计算距离矩阵 (家庭 x 销售点)
    household_points = household_coords[['long', 'lat']].values
    sales_points = sales_coords[['lon', 'lat']].values
    
    distance_matrix = cdist(household_points, sales_points, metric='euclidean')
    
    print("开始为每个销售点分配家庭（一一对应）...")
    
    # 贪心算法：为每个销售点找最近的未分配家庭
    household_to_sales_mapping = {}
    used_households = set()
    assigned_sales_count = 0
    
    # 创建所有可能的(销售点索引, 家庭索引, 距离)组合
    all_pairs = []
    for sales_idx in range(len(sales_coords)):
        for household_idx in range(len(household_coords)):
            distance = distance_matrix[household_idx, sales_idx]
            all_pairs.append((sales_idx, household_idx, distance))
    
    # 按距离排序
    all_pairs.sort(key=lambda x: x[2])
    
    assigned_sales_points = set()
    
    # 第一轮：一一对应分配
    print("第一轮：为每个销售点分配唯一家庭...")
    for sales_idx, household_idx, distance in all_pairs:
        # 如果这个销售点还没分配，且这个家庭还没被使用
        if sales_idx not in assigned_sales_points and household_idx not in used_households:
            household_id = household_coords.iloc[household_idx]['hhold']
            sales_coord = sales_coords.iloc[sales_idx]
            
            household_to_sales_mapping[household_id] = {
                'lon': sales_coord['lon'],
                'lat': sales_coord['lat']
            }
            
            used_households.add(household_idx)
            assigned_sales_points.add(sales_idx)
            assigned_sales_count += 1
            
            print(f"销售点 {sales_idx} -> 家庭 {household_id}, 距离: {distance:.6f}")
    
    print(f"第一轮完成：分配了 {assigned_sales_count} 个销售点")
    
    # 第二轮：为剩余销售点分配家庭（允许重复）
    if len(assigned_sales_points) < len(sales_coords):
        print("第二轮：为剩余销售点分配家庭（允许重复使用家庭）...")
        remaining_sales = set(range(len(sales_coords))) - assigned_sales_points
        
        for sales_idx in remaining_sales:
            # 找到距离这个销售点最近的家庭
            distances_to_sales = distance_matrix[:, sales_idx]
            nearest_household_idx = np.argmin(distances_to_sales)
            
            household_id = household_coords.iloc[nearest_household_idx]['hhold']
            sales_coord = sales_coords.iloc[sales_idx]
            
            # 为多个家庭ID创建唯一标识
            unique_household_id = f"{household_id}_copy_{sales_idx}"
            household_to_sales_mapping[unique_household_id] = {
                'lon': sales_coord['lon'],
                'lat': sales_coord['lat']
            }
            
            assigned_sales_count += 1
            print(f"销售点 {sales_idx} -> 家庭 {unique_household_id} (复用 {household_id}), 距离: {distances_to_sales[nearest_household_idx]:.6f}")
    
    print(f"\n最终结果：")
    print(f"成功分配的销售点数量: {assigned_sales_count}")
    print(f"分配的家庭数量: {len(household_to_sales_mapping)}")
    print(f"是否所有销售点都被分配: {assigned_sales_count == len(sales_coords)}")
    
    return household_to_sales_mapping

def update_pop_subset_coordinates(pop_subset, household_mapping):
    """
    根据匹配结果更新pop_subset中的坐标
    """
    pop_subset_updated = pop_subset.copy()
    
    for household_id, new_coords in household_mapping.items():
        mask = pop_subset_updated['hhold'] == household_id
        pop_subset_updated.loc[mask, 'long'] = new_coords['lon']
        pop_subset_updated.loc[mask, 'lat'] = new_coords['lat']
        # 更新geometry列
        pop_subset_updated.loc[mask, 'geometry'] = f"POINT ({new_coords['lon']} {new_coords['lat']})"
    
    return pop_subset_updated

# 执行坐标匹配
print("开始进行家庭与销售点的坐标匹配...")
household_mapping = match_households_to_sales_locations(pop_subset, sales_df_filtered)

# 更新pop_subset的坐标
pop_subset_updated = update_pop_subset_coordinates(pop_subset, household_mapping)

print(f"\n匹配完成！共匹配了 {len(household_mapping)} 个家庭")
print(f"原始pop_subset数据量: {len(pop_subset)}")
print(f"更新后pop_subset数据量: {len(pop_subset_updated)}")

# 显示更新后的数据
print("\n更新后的pop_subset样本数据:")
print(pop_subset_updated.head(10))

pop_subset_updated.to_csv('pop_subset_updated.csv')
