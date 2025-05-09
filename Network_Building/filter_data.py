import pandas as pd

# 1. Read the dataset.
#    - Assume columns are in the order: UserID, Latitude, Longitude, LocationID, Timestamp
#    - Assume values are separated by whitespace and no header row in the file.
df = pd.read_csv(
    "Gowalla_totalCheckins.txt",
    sep=r"\s+",
    header=None,
    names=["UserID", "Timestamp","Latitude", "Longitude", "LocationID" ]
)

# 2. Define the bounding box for lat/lon
# min_lat, max_lat = 40.55, 40.95
# min_lon, max_lon = -73.70, -72.80

# NYC
min_lat, max_lat = 40.65, 40.78
min_lon, max_lon = -74.05, -73.86

# 3. Filter rows based on the bounding box
filtered_df = df[
    (df["Latitude"] >= min_lat) & (df["Latitude"] <= max_lat) &
    (df["Longitude"] >= min_lon) & (df["Longitude"] <= max_lon)
]

# 4. Write the filtered rows to a new text file
filtered_df.to_csv(
    "Gowalla_filtered_nyc.txt",
    sep="\t",      # or use sep=" " if you prefer space-separated
    index=False,
    header=False   # if you want no header line in the output
)

print("Filtering complete! Saved filtered data to Gowalla_filtered.txt.")
