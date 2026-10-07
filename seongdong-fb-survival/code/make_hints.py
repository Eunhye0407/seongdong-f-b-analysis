"""
동 키워드로 수집한 seongdong_group*_restaurants.csv 에서
not_found 점포와 같은 건물 + 같은 이름인 네이버 매장을 찾아 place_hints.csv 생성

사용법: python3 make_hints.py cohort_restaurants_clean.csv
"""
import glob
import sys
import pandas as pd
from naver_crawling_cohort import addr_key, name_sim

coh = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "cohort_restaurants_clean.csv",
                  encoding="utf-8-sig", dtype={"place_id": str})
miss = coh[~coh["match_status"].astype(str).str.startswith("matched")].copy()
miss["ak"] = miss["도로명주소"].map(addr_key)

files = sorted(glob.glob("seongdong_group*_restaurants.csv"))
g = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype={"place_id": str}) for f in files], ignore_index=True)
g = g.dropna(subset=["naver_address"]).drop_duplicates("place_id")
g["ak"] = g["naver_address"].map(addr_key)

rows = []
for r in miss.itertuples():
    cand = g[g["ak"] == r.ak]
    best = max(((name_sim(r.사업장명, q.name), q.place_id, q.name) for q in cand.itertuples()), default=None)
    if best and best[0] >= 0.8:
        rows.append({"restaurant_id": r.restaurant_id, "place_id": best[1], "사업장명": r.사업장명, "naver_name": best[2]})
out = pd.DataFrame(rows)
out.to_csv("place_hints.csv", index=False, encoding="utf-8-sig")
print(f"동 키워드 파일 {len(files)}개 | 미매칭 {len(miss)}개 중 힌트 {len(out)}개 -> place_hints.csv")
