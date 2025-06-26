import pandas as pd

sales_df=pd.read_csv("data/filtered_sales_data.csv")

date_cols = ['S_1DATE', 'S_2DATE', 'S_3DATE', 'S_4DATE']
start_threshold = pd.Timestamp("2022-07-01", tz="UTC")  # Start time threshold
end_threshold = pd.Timestamp("2023-10-01", tz="UTC")    # End time threshold

def get_first_sale_in_period(row):
    # Parse all dates as datetime (with timezone)
    dates = pd.to_datetime(row[date_cols], errors='coerce', utc=True)
    # Filter dates within the specified time range
    filtered = dates[(dates > start_threshold) & (dates < end_threshold)].sort_values()
    # If there are results, format as string, otherwise return '0'
    return filtered.iloc[0].strftime("%Y-%m-%d") if not filtered.empty else '0'

# Manual progress display implementation
print("Processing sales data...")
total_rows = len(sales_df)
results = []

for i, row in enumerate(sales_df.iterrows()):
    if i % 1000 == 0:  # Show progress every 1000 rows
        progress = (i / total_rows) * 100
        print(f"Progress: {progress:.1f}% ({i}/{total_rows})")
    
    result = get_first_sale_in_period(row[1])
    results.append(result)

sales_df['first_sale_in_period'] = results
print("Processing completed!")

# Keep only longitude, latitude and processing results
result = sales_df[['lon', 'lat', 'first_sale_in_period']]
result.to_csv("data/sales_data_with_first_recent_sale.csv", index=False)
print(result)