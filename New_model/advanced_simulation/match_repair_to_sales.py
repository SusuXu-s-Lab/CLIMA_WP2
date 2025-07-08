import pandas as pd
import numpy as np

def calculate_distance(coord1, coord2):
    """计算两点间的欧几里得距离"""
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def match_repair_to_sales():
    """将repair中的每个点分配给sales中的唯一点，不允许重复"""
    
    # 定义边界框
    bbox = (-81.97581623909956, 26.504952873345545, -81.95459566656734, 26.52311972089953)
    minx, miny, maxx, maxy = bbox
    
    try:
        # 读取数据
        print("读取和过滤数据...")
        sales_df = pd.read_csv("fl/fl/sales_data_with_recent_sale.csv")
        repair_df = pd.read_csv("fl/fl/merged_with_coordinates.csv")
        
        # 应用边界框过滤
        sales_df_filtered = sales_df[
            (sales_df['lon'] >= minx) & (sales_df['lon'] <= maxx) &
            (sales_df['lat'] >= miny) & (sales_df['lat'] <= maxy)
        ]
        
        repair_df_filtered = repair_df[
            (repair_df['lon'] >= minx) & (repair_df['lon'] <= maxx) &
            (repair_df['lat'] >= miny) & (repair_df['lat'] <= maxy)
        ]
        
        # 获取唯一坐标
        sales_coords = sales_df_filtered[['lon', 'lat']].drop_duplicates().reset_index(drop=True)
        repair_coords = repair_df_filtered[['lon', 'lat']].drop_duplicates().reset_index(drop=True)
        
        print(f"Sales唯一坐标数量: {len(sales_coords)}")
        print(f"Repair唯一坐标数量: {len(repair_coords)}")
        
        if len(repair_coords) > len(sales_coords):
            print("错误：Repair坐标数量超过Sales坐标数量，无法实现一一对应")
            return None
        
        print("\n开始计算距离矩阵...")
        # 计算距离矩阵 (repair x sales)
        distance_matrix = np.zeros((len(repair_coords), len(sales_coords)))
        
        for repair_idx in range(len(repair_coords)):
            repair_coord = [repair_coords.iloc[repair_idx]['lon'], repair_coords.iloc[repair_idx]['lat']]
            for sales_idx in range(len(sales_coords)):
                sales_coord = [sales_coords.iloc[sales_idx]['lon'], sales_coords.iloc[sales_idx]['lat']]
                distance_matrix[repair_idx, sales_idx] = calculate_distance(repair_coord, sales_coord)
        
        print("距离矩阵计算完成")
        
        # 贪心匹配算法：为每个repair点找到最近的未分配sales点
        print("\n开始匹配过程...")
        
        # 创建所有可能的(repair索引, sales索引, 距离)组合
        all_pairs = []
        for repair_idx in range(len(repair_coords)):
            for sales_idx in range(len(sales_coords)):
                distance = distance_matrix[repair_idx, sales_idx]
                all_pairs.append((repair_idx, sales_idx, distance))
        
        # 按距离排序
        all_pairs.sort(key=lambda x: x[2])
        
        # 执行匹配
        repair_to_sales_mapping = {}
        used_sales_indices = set()
        assigned_repair_count = 0
        
        print("开始分配最优匹配...")
        for repair_idx, sales_idx, distance in all_pairs:
            # 如果这个sales点还没被分配，且这个repair点还没被匹配
            if sales_idx not in used_sales_indices and repair_idx not in repair_to_sales_mapping:
                repair_coord = repair_coords.iloc[repair_idx]
                sales_coord = sales_coords.iloc[sales_idx]
                
                repair_to_sales_mapping[repair_idx] = {
                    'sales_idx': sales_idx,
                    'repair_lon': repair_coord['lon'],
                    'repair_lat': repair_coord['lat'],
                    'sales_lon': sales_coord['lon'],
                    'sales_lat': sales_coord['lat'],
                    'distance': distance,
                    'distance_meters': distance * 111320  # 转换为米
                }
                
                used_sales_indices.add(sales_idx)
                assigned_repair_count += 1
                
                # 打印前10个匹配
                if assigned_repair_count <= 10:
                    print(f"Repair {repair_idx}: ({repair_coord['lon']:.6f}, {repair_coord['lat']:.6f}) -> "
                          f"Sales {sales_idx}: ({sales_coord['lon']:.6f}, {sales_coord['lat']:.6f}), "
                          f"距离: {distance:.6f}度 ({distance * 111320:.1f}米)")
        
        print(f"\n匹配完成！")
        print(f"成功匹配的repair点数量: {assigned_repair_count}")
        print(f"使用的sales点数量: {len(used_sales_indices)}")
        print(f"剩余未使用的sales点数量: {len(sales_coords) - len(used_sales_indices)}")
        
        # 统计距离分布
        distances = [mapping['distance_meters'] for mapping in repair_to_sales_mapping.values()]
        print(f"\n距离统计:")
        print(f"最小距离: {min(distances):.1f}米")
        print(f"最大距离: {max(distances):.1f}米")
        print(f"平均距离: {np.mean(distances):.1f}米")
        print(f"中位距离: {np.median(distances):.1f}米")
        
        # 创建结果DataFrame
        result_data = []
        for repair_idx, mapping in repair_to_sales_mapping.items():
            result_data.append({
                'repair_idx': repair_idx,
                'repair_lon': mapping['repair_lon'],
                'repair_lat': mapping['repair_lat'],
                'sales_idx': mapping['sales_idx'],
                'sales_lon': mapping['sales_lon'],
                'sales_lat': mapping['sales_lat'],
                'distance_degrees': mapping['distance'],
                'distance_meters': mapping['distance_meters']
            })
        
        result_df = pd.DataFrame(result_data)
        
        # 保存结果
        result_df.to_csv("fl/fl/repair_to_sales_mapping.csv", index=False)
        print(f"\n结果已保存到 fl/fl/repair_to_sales_mapping.csv")
        
        # 创建完整的映射后repair数据文件（保留所有原始信息，只更新坐标）
        print("\n创建完整的映射后repair数据...")
        
        # 首先创建坐标到原始数据的映射
        coord_to_original_data = {}
        for idx, row in repair_df_filtered.iterrows():
            coord_key = f"{row['lon']:.6f},{row['lat']:.6f}"
            if coord_key not in coord_to_original_data:
                coord_to_original_data[coord_key] = []
            coord_to_original_data[coord_key].append(row)
        
        # 创建映射后的完整数据
        mapped_repair_data = []
        
        for repair_idx, mapping in repair_to_sales_mapping.items():
            original_coord = repair_coords.iloc[repair_idx]
            coord_key = f"{original_coord['lon']:.6f},{original_coord['lat']:.6f}"
            
            # 获取这个坐标对应的所有原始记录
            if coord_key in coord_to_original_data:
                for original_record in coord_to_original_data[coord_key]:
                    # 复制原始记录的所有信息
                    mapped_record = original_record.copy()
                    # 只更新坐标为对应的sales坐标
                    mapped_record['lon'] = mapping['sales_lon']
                    mapped_record['lat'] = mapping['sales_lat']
                    # 添加映射信息作为新列
                    mapped_record['original_lon'] = mapping['repair_lon']
                    mapped_record['original_lat'] = mapping['repair_lat']
                    mapped_record['mapping_distance_meters'] = mapping['distance_meters']
                    mapped_record['mapped_to_sales_idx'] = mapping['sales_idx']
                    
                    mapped_repair_data.append(mapped_record)
        
        # 转换为DataFrame
        mapped_repair_df = pd.DataFrame(mapped_repair_data)
        
        # 保存完整的映射后数据
        mapped_repair_df.to_csv("fl/fl/repair_coords_mapped_to_sales.csv", index=False)
        print(f"映射后的完整repair数据已保存到 fl/fl/repair_coords_mapped_to_sales.csv")
        print(f"原始repair_df_filtered记录数: {len(repair_df_filtered)}")
        print(f"映射后数据记录数: {len(mapped_repair_df)}")
        print(f"数据列数: {len(mapped_repair_df.columns)}")
        print(f"新增列: original_lon, original_lat, mapping_distance_meters, mapped_to_sales_idx")
        
        # 分析距离分布
        print(f"\n距离分布分析:")
        bins = [0, 50, 100, 200, 500, 1000, float('inf')]
        labels = ['0-50m', '50-100m', '100-200m', '200-500m', '500-1000m', '>1000m']
        
        for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
            count = sum(1 for d in distances if lower <= d < upper)
            percentage = count / len(distances) * 100
            print(f"{labels[i]}: {count} 个点 ({percentage:.1f}%)")
        
        return repair_to_sales_mapping
        
    except Exception as e:
        print(f"错误: {e}")
        return None

if __name__ == "__main__":
    print("开始将Repair坐标匹配到Sales坐标...\n")
    mapping = match_repair_to_sales()
    
    if mapping:
        print(f"\n✅ 匹配成功完成！")
        print(f"📊 总计匹配了 {len(mapping)} 个repair点到唯一的sales点")
        print(f"🎯 每个repair点都有唯一对应的sales点，无重复分配")
    else:
        print("❌ 匹配失败") 