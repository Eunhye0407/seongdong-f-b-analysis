"""팀원들이 나눠 돌린 part 파일을 하나로 합치기:  python3 combine_parts.py"""
import glob
import pandas as pd

for kind in ["restaurants", "reviews"]:
    files = sorted(glob.glob(f"cohort_part*_{kind}.csv"))
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype={"place_id": str}) for f in files], ignore_index=True)
    key = "restaurant_id" if kind == "restaurants" else "review_id"
    df = df.drop_duplicates(key)
    df.to_csv(f"cohort_{kind}_all.csv", index=False, encoding="utf-8-sig")
    print(f"{kind}: {len(files)}개 파일 -> {len(df):,}행 (cohort_{kind}_all.csv)")
