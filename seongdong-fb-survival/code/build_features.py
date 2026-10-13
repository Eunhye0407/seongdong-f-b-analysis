"""
분석용 최종 데이터셋 생성: 열 이름 정리 + 임대료 범주 + 반경 경쟁 변수 + 경쟁 네트워크 변수

입력  data/interim/restaurants_with_reviews_raw.csv
      (notebooks/preprocessing_submission.ipynb 결과에 code/review_merge.py로 리뷰 요약을 붙인 파일)
출력  data/final_dataset_with_reviews.csv   점포 1행, 3,775행
      outputs/tables/variable_summary.csv  주요 변수 기술통계

실행  python code/build_features.py

변수 정의 (반경 r = 100, 200, 300, 500m, 기준일 2021-04-01에 영업 중인 이웃 기준)
  competitor_c_{r}m   같은 업태 이웃 수                       (Competitor_C)
  adjacent_biz_{r}m   업태와 무관한 전체 음식점 이웃 수         (Adjacent_BIZ)
  competitor_r_{r}m   같은 업태 비율 = C / BIZ (BIZ=0이면 0)  (Competitor_R)
  net_strength_{r}m   가중 연결강도: 같은 업태 이웃의 Σ(100 / 거리m), 100m 이웃 1곳 = 1
  net_betweenness_{r}m 매개중심성: 거리를 엣지 길이로 둔 최단경로 기준, 정규화(0~1)
  net_clustering_{r}m 군집계수: 같은 업태 이웃끼리도 서로 이웃인 비율(이웃 2곳 미만이면 0)

경쟁 네트워크: 노드 = 점포, 엣지 = 반경 r 안의 같은 업태 점포 쌍, 가중치 = 100 / 거리
  - 거리는 EPSG:5186(중부원점) 평면좌표로 계산한다.
  - 같은 건물(거리 0m) 쌍이 많으므로 거리는 최소 10m로 둔다.
  - 업종 코드가 없는 점포는 '미분류'를 하나의 업태로 본다(기존 competitors_same_300m 정의와 동일).
"""
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "interim" / "restaurants_with_reviews_raw.csv"
OUT_PATH = ROOT / "data" / "final_dataset_with_reviews.csv"
TABLE_DIR = ROOT / "outputs" / "tables"

RADII = [100, 200, 300, 500]
MIN_DIST_M = 10            # 같은 건물 점포의 거리 하한
UNMAPPED = "미분류"

# 임대료(동 단위 중위값, 평당 전세가) 범주: 동 값 사이의 큰 간격(1,364→2,064, 2,633→3,044)을 경계로 3단계
RENT_BINS = [0, 1700, 2900, np.inf]
RENT_LABELS = ["low", "mid", "high"]

# 원자료 열 → 분석용 열 이름
RENAME = {
    "restaurant_id": "restaurant_id",
    "사업장명": "store_name",
    "인허가일자": "license_date",
    "폐업일자": "closure_date",
    "지번주소": "address_jibun",
    "longitude": "longitude",
    "latitude": "latitude",
    "서울시_서비스_업종_코드": "category_code",
    "서울시_서비스_업종_코드_명": "category_name",
    "TRDAR_CD": "trdar_code",
    "TRDAR_CD_N": "trdar_name",
    "area_category_sales_per_store_final": "area_sales_per_store",
    "rent_dong": "dong",
    "rent_median": "rent_median",
    "business_age_at_baseline_years": "business_age_years",
    "event": "event",
    "observation_end": "observation_end",
    "duration_days": "duration_days",
    "place_id": "naver_place_id",
    "match_status": "review_match_status",
    "shared_place": "review_shared_place",
    "has_review": "has_review",
    "review_volume": "review_volume",
    "review_segment": "review_segment",
    "review_segment_n": "review_tagged_n",
    "review_revisit_share": "review_revisit_share",
}
# 사용하지 않는 원자료 열: rent_mean(중위값만 사용), duration_months(일 단위와 중복),
# restaurants_all_300m / competitors_same_300m(아래에서 반경별로 다시 계산)


def load_raw() -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    df = raw[list(RENAME)].rename(columns=RENAME)
    df["trdar_code"] = df["trdar_code"].astype("Int64")
    df["naver_place_id"] = df["naver_place_id"].astype("Int64")
    return df


def add_rent_level(df: pd.DataFrame) -> pd.DataFrame:
    df["rent_level"] = pd.cut(df["rent_median"], bins=RENT_BINS, labels=RENT_LABELS).astype(str)
    # 한 동이 두 범주에 걸치지 않는지 확인
    assert df.groupby("dong")["rent_level"].nunique().max() == 1
    return df


def pairwise_distance(df: pd.DataFrame) -> np.ndarray:
    """점포 간 거리 행렬(m). 자기 자신은 inf."""
    to_tm = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
    x, y = to_tm.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    dist = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(dist, np.inf)
    return dist


def add_radius_competition(df: pd.DataFrame, dist: np.ndarray, same: np.ndarray) -> pd.DataFrame:
    for r in RADII:
        within = dist <= r
        c = (within & same).sum(axis=1)
        biz = within.sum(axis=1)
        df[f"competitor_c_{r}m"] = c
        df[f"adjacent_biz_{r}m"] = biz
        df[f"competitor_r_{r}m"] = np.divide(c, biz, out=np.zeros(len(df)), where=biz > 0)
    return df


def add_network(df: pd.DataFrame, dist: np.ndarray, same: np.ndarray) -> pd.DataFrame:
    n = len(df)
    for r in RADII:
        i, j = np.where(np.triu((dist <= r) & same, k=1))
        d = np.maximum(dist[i, j], MIN_DIST_M)
        g = ig.Graph(n=n, edges=list(zip(i.tolist(), j.tolist())), directed=False)
        g.es["length"] = d                 # 최단경로용 거리
        g.es["weight"] = 100.0 / d         # 연결강도용 가중치

        df[f"net_strength_{r}m"] = g.strength(weights="weight")
        btw = np.asarray(g.betweenness(weights="length", directed=False))
        df[f"net_betweenness_{r}m"] = btw / ((n - 1) * (n - 2) / 2)
        df[f"net_clustering_{r}m"] = np.nan_to_num(g.transitivity_local_undirected(mode="zero"))
        print(f"  {r:>3}m 네트워크: 엣지 {g.ecount():,}개, 고립 노드 {sum(1 for v in g.degree() if v == 0):,}개")
    return df


def check(df: pd.DataFrame, raw_path: Path) -> None:
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    assert df["restaurant_id"].is_unique and len(df) == len(raw)
    for r_small, r_big in zip(RADII, RADII[1:]):
        assert (df[f"adjacent_biz_{r_big}m"] >= df[f"adjacent_biz_{r_small}m"]).all()
    assert (df["competitor_c_300m"] <= df["adjacent_biz_300m"]).all()
    # 원자료의 300m 변수(다른 좌표계로 계산)와 일치 정도
    m_all = (df["adjacent_biz_300m"] == raw["restaurants_all_300m"]).mean()
    m_same = (df["competitor_c_300m"] == raw["competitors_same_300m"]).mean()
    corr = np.corrcoef(df["adjacent_biz_300m"], raw["restaurants_all_300m"])[0, 1]
    print(f"  원자료 300m 변수와 정확히 일치: 전체 {m_all:.1%}, 동종 {m_same:.1%} (상관 {corr:.4f}; 나머지는 경계의 ±1 차이)")


def main() -> None:
    print("1) 원자료 로드 및 열 이름 정리")
    df = load_raw()

    print("2) 임대료 범주(rent_level)")
    df = add_rent_level(df)
    print(df.groupby("rent_level")["dong"].unique().to_string())

    print("3) 반경 경쟁 변수")
    dist = pairwise_distance(df)
    cat = df["category_code"].fillna(UNMAPPED).to_numpy(dtype=object)
    same = cat[:, None] == cat[None, :]
    df = add_radius_competition(df, dist, same)

    print("4) 경쟁 네트워크 변수")
    df = add_network(df, dist, same)

    print("5) 검증")
    check(df, RAW_PATH)

    base = list(RENAME.values())
    base.insert(base.index("rent_median") + 1, "rent_level")
    cut = base.index("naver_place_id")                     # 생존 변수 뒤, 리뷰 변수 앞에 경쟁 변수 배치
    competition = [c for c in df.columns if c.startswith(("competitor_", "adjacent_", "net_"))]
    df = df[base[:cut] + competition + base[cut:]]
    # 저장 전 소수 자릿수 정리 (매개중심성은 값이 매우 작아 유효숫자를 남김)
    digits = {c: 6 for c in df.select_dtypes("float").columns}
    digits.update({"longitude": 7, "latitude": 7, **{c: 10 for c in df.columns if c.startswith("net_betweenness")}})
    df = df.round(digits)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    num = df.select_dtypes("number").drop(columns=["trdar_code", "naver_place_id", "longitude", "latitude"])
    summary = num.describe().T[["count", "mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median"})
    summary.round(6).to_csv(TABLE_DIR / "variable_summary.csv", encoding="utf-8-sig")
    print(f"저장: {OUT_PATH.relative_to(ROOT)}  ({len(df):,}행 × {df.shape[1]}열)")


if __name__ == "__main__":
    main()
