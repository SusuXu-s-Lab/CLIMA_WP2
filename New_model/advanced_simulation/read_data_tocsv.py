import pyogrio

# bbox = (-82.04150698533131, 26.604200914701607, -81.87502336164845, 26.490448860026532)   # (minx, miny, maxx, maxy)
bbox = (-81.97581623909956, 26.504952873345545, -81.95459566656734, 26.52311972089953)   # (minx, miny, maxx, maxy)


pop_subset = pyogrio.read_dataframe(
    "fl/fl/fl_population.gpkg",
    layer="fl_population",
    bbox=bbox
)
pop_subset.to_csv("pop_subset.csv", index=False)

wp_subset = pyogrio.read_dataframe(
    "fl/fl/fl_workplace.gpkg",
    layer="fl_workplace",
    bbox=bbox
)
wp_subset.to_csv("wp_subset.csv", index=False)


print('pop_subset:\n', pop_subset)
print('wp_subset:\n', wp_subset)
