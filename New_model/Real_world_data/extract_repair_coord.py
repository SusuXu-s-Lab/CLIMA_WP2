import asyncio, aiohttp, csv, pandas as pd
import sys
import os
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"

async def geocode(session, address, sem):
    async with sem:                                   # 在同一 loop 内使用
        params = dict(address=address, benchmark="4", format="json")
        async with session.get(URL, params=params, timeout=10) as resp:
            data = await resp.json()
            matches = data.get("result", {}).get("addressMatches", [])
            if matches:
                m = matches[0]
                c = m["coordinates"]
                return address, m["matchedAddress"], c["x"], c["y"]
    return address, None, None, None

async def main():
    df = pd.read_csv("data/RecordList202309.csv")
    cities = {'FORT MYERS','LEHIGH ACRES','BOKEELIA',
              'NORTH FORT MYERS','MATLACHA','FORT MYERS BEACH'}
    addrs = [a for a in df['Address'] if isinstance(a,str) and
             any(c in a.upper() for c in cities)]

    sem = asyncio.Semaphore(10)
    results = []

    async with aiohttp.ClientSession() as sess:
        tasks = [asyncio.create_task(geocode(sess, a, sem)) for a in addrs]
        for coro in asyncio.as_completed(tasks):
            addr, matched, lon, lat = await coro
            if matched:
                print(f"✅ {addr} → {matched} @ ({lat}, {lon})")
                results.append((addr, matched, lon, lat))
            else:
                print(f"❌ No result for {addr}")

    # ---------- 追加写入，不覆盖旧内容 ----------
    file_path = "single_geocode_result.csv"
    write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["address", "matched_address", "lon", "lat"])
        writer.writerows(results)

if __name__ == "__main__":
    asyncio.run(main())
