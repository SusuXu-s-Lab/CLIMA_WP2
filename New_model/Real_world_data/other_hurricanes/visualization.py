import requests
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pdb

# ========== Step 1: Crawl Data ==========
# dates = pd.date_range("2017-06", "2018-12", freq="MS")
# city='Houston'

dates = pd.date_range("2017-07", "2018-12", freq="MS")
city='Miami'

# dates = pd.date_range("2005-06", "2006-12", freq="MS") records = [] 
# city='New Orleans'

records = []

for date in dates:
    url = f"https://www.census.gov/construction/bps/txt/tb3u{date.year}{date.month:02d}.txt"
    r = requests.get(url)
    if r.status_code != 200:
        continue
    text = r.text
    match = re.search(rf"{city}[^\n]*?LA[^\d]*(\d+)", text, re.IGNORECASE)
    if match:
        permits = int(match.group(1))
        records.append({"date": date, "permits": permits})

df = pd.DataFrame(records).sort_values("date")

# ========== Step 2: Visualization ==========
plt.figure(figsize=(14, 4.5))  
plt.plot(df["date"], df["permits"], 
         marker='o', color="#0077B6", linewidth=2.5, markersize=6, 
         markerfacecolor="#D81B60", markeredgecolor="white")


for x, y in zip(df["date"], df["permits"]):
    plt.text(x, y + 10, f"{y}", ha='center', va='bottom', fontsize=9, color="#0077B6")


plt.title(f"Monthly Building Permits in {city} {dates[0].strftime('%b %Y')} - {dates[-1].strftime('%b %Y')}", fontsize=14, fontweight="bold")
plt.ylabel("Permits", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.grid(alpha=0.25)
plt.xticks(rotation=35, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(f"images/{city}.png")
plt.show()
