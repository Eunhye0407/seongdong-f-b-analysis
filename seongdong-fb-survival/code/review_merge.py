"""
네이버 리뷰 -> 리뷰 요약 3열 -> final_restaurant_level_dataset.csv 결합

  review_volume        : log(1 + 방문자 리뷰 총수)                 얼마나 많이 찾나 (수요)
  review_segment       : 리뷰 태그의 주된 동행·목적 (연인/친구/가족/업무/혼자)  누가 오나 (손님층)
  review_revisit_share : 'N번째 방문'이 2 이상인 리뷰 비율              단골이 있나 (충성도)

사용법 (part 파일들을 종류별로 하나로 합친 뒤):
    python3 review_merge.py cohort_reviews_all.csv cohort_restaurants_all.csv final_restaurant_level_dataset.csv
    (예전 동 키워드 방식 파일도 그대로 넣으면 자동으로 주소+상호명 매칭)
"""
import re
import sys
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

MIN_TAGGED = 3      # segment 판정에 필요한 최소 태그 리뷰 수
MIN_VISIT = 5       # revisit_share 계산에 필요한 최소 리뷰 수

# ------------------------------------------------------------
# 리뷰 카드 태그 -> 손님층
#   실제 구조: ...저녁에 방문예약 없이 이용대기 시간 바로 입장데이트연인・배우자, 친구 음식이 맛있어요...
#   방문시간·예약·대기 정보 바로 뒤(붙어 있거나 한 칸 띄움)에 [방문목적][동행자]가 나옴.
#   네이버가 정해 둔 선택지 단어만 인정 -> 닉네임·키워드 칩('아이와 가기 좋아요')은 걸리지 않음
# ------------------------------------------------------------
ANCHOR = r"에 방문|예약 없이 이용|예약 후 이용|바로 입장|\d+\s*(?:분|시간)\s*(?:이내|이상)"

# 성동구 1그룹 실제 데이터에서 확인한 선택지
PURPOSE_VOCAB = ["가족모임", "포장·배달", "비즈니스", "데이트", "나들이", "기념일", "일상", "친목", "회식", "여행", "기타"]
COMPANION_VOCAB = ["친척·형제자매", "연인·배우자", "지인·동료", "반려동물", "부모님", "친구", "혼자", "아이", "기타"]

# 'lift' : 성동구 전체 평균 대비 이 가게에서 특히 많이 나온 손님층 (기본값, 연인·친구 쏠림 방지)
# 'mode' : 가장 많이 나온 손님층
SEGMENT_METHOD = "lift"

_alt = lambda v: "|".join(re.escape(x) for x in sorted(v, key=len, reverse=True))
_P, _C = _alt(PURPOSE_VOCAB), _alt(COMPANION_VOCAB)
TAG_RE = re.compile(rf"((?:{_P})(?:,\s*(?:{_P}))*)?((?:{_C})(?:,\s*(?:{_C}))*)?(?=\s|\||$)")


def parse_tags(meta):
    """-> (목적 리스트, 동행자 리스트) / 태그 없으면 None"""
    t = re.sub(r"[・•ㆍ･]", "·", str(meta)).split("반응 남기기")[0]
    last = None
    for m in re.finditer(ANCHOR, t):
        last = m
    if last is None: return None
    rest = t[last.end():].lstrip(" |")
    m = TAG_RE.match(rest)
    if not m or not m.group(0): return None
    split = lambda s: [x.strip() for x in s.split(",")] if s else []
    return split(m.group(1)), split(m.group(2))

def tag_chunk(meta):
    """점검 출력용 문자열: '데이트 / 연인·배우자, 친구'"""
    p = parse_tags(meta)
    return None if p is None else f"{', '.join(p[0]) or '-'} / {', '.join(p[1]) or '-'}"

def review_tag(meta):
    p = parse_tags(meta)
    if p is None: return None
    purposes, comps = set(p[0]), set(p[1])
    # 1순위: 누구와
    if comps & {"아이", "부모님", "친척·형제자매"}: return "가족"     # '연인·배우자, 아이' = 가족 외식
    if "연인·배우자" in comps: return "연인"
    if "친구" in comps: return "친구"
    if "지인·동료" in comps:
        return "업무" if purposes & {"회식", "비즈니스"} else "친구"   # 친목·일상으로 만난 지인은 친구로
    if "혼자" in comps: return "혼자"
    # 2순위: 동행자 태그가 없거나(반려동물·기타) 모호하면 목적으로 추정
    if "데이트" in purposes: return "연인"
    if "가족모임" in purposes: return "가족"
    if purposes & {"회식", "비즈니스"}: return "업무"
    if "친목" in purposes: return "친구"
    return None                                                     # 일상·나들이·여행·기념일만 있으면 판정 안 함


def pick_segment(counts, overall):
    if counts.sum() < MIN_TAGGED: return np.nan
    if SEGMENT_METHOD == "mode": return counts.idxmax()
    lift = (counts / counts.sum()) / overall.reindex(counts.index)
    lift = lift[counts >= 2]                                         # 1건짜리 우연은 제외
    return lift.idxmax() if len(lift) else counts.idxmax()

def summarize_reviews(rv):
    rv = rv.drop_duplicates("review_id").copy()
    rv["segment"] = rv["meta_text"].map(review_tag)
    rv["visit_count"] = pd.to_numeric(rv["visit_count"], errors="coerce")
    overall = rv["segment"].value_counts(normalize=True)
    out = []
    for pid, g in rv.groupby("place_id"):
        counts = g["segment"].value_counts()
        vc = g["visit_count"].dropna()
        out.append({
            "place_id": pid,
            "review_segment": pick_segment(counts, overall),
            "review_segment_n": int(counts.sum()),
            "review_revisit_share": (vc >= 2).mean() if len(vc) >= MIN_VISIT else np.nan,
        })
    return pd.DataFrame(out), rv


# ------------------------------------------------------------
# 2. 매칭: 도로명+건물번호로 건물 -> 그 안에서 상호명
# ------------------------------------------------------------
def addr_key(a):
    m = re.search(r"([가-힣0-9]+(?:로|길))\s*(\d+(?:-\d+)?)", str(a))
    return f"{m.group(1)}{m.group(2)}" if m else None

def norm_name(x):
    x = re.sub(r"\(.*?\)|\[.*?\]|주식회사|\(주\)|㈜", "", str(x).lower())
    return re.sub(r"[^0-9a-z가-힣]", "", x)

def name_sim(a, b):
    if not a or not b: return 0.0
    if a == b: return 1.0
    if a in b or b in a: return 0.9
    return SequenceMatcher(None, a, b).ratio()

def match_places(pi, ds):
    d_addr = ds["도로명주소"].map(addr_key).to_numpy()
    d_nm = ds["사업장명"].map(norm_name).to_numpy()
    rows = []
    for p in pi.itertuples(index=False):
        pa, pn = addr_key(p.naver_address), norm_name(p.name)
        if not pa: continue
        cand = [(name_sim(pn, d_nm[i]), i) for i in np.where(d_addr == pa)[0]]
        cand = [c for c in cand if c[0] >= 0.5]
        if cand:
            sc, i = max(cand)
            rows.append({"place_id": p.place_id, "restaurant_id": ds["restaurant_id"].iat[i], "match_score": round(sc, 3)})
    m = pd.DataFrame(rows, columns=["place_id", "restaurant_id", "match_score"])
    return m.sort_values("match_score", ascending=False).drop_duplicates("restaurant_id")


# ------------------------------------------------------------
# 3. 결합
# ------------------------------------------------------------
def main(review_path, place_path, ds_path):
    rd = lambda p: pd.read_csv(p, encoding="utf-8-sig", dtype={"place_id": str})
    rv, pi = rd(review_path), rd(place_path)
    ds = pd.read_csv(ds_path, encoding="utf-8-sig")

    if "restaurant_id" in pi.columns:
        # 대상 점포 직접 검색 방식(naver_crawling_cohort.py): 크롤링 때 주소로 확인 완료 -> 바로 연결
        print("[점포 직접 검색 결과] match_status")
        print(pi["match_status"].value_counts().to_string())
        pi = pi[pi["match_status"].astype(str).str.startswith("matched")].drop_duplicates("restaurant_id")
        keep = [c for c in ["place_id", "restaurant_id", "match_status", "shared_place"] if c in pi.columns]
        m = pi[keep]                                                    # 한 매장에 인허가 여러 건이면 모두 연결
    else:
        # 동 키워드 검색 방식: 도로명주소 + 상호명으로 매칭
        pi = pi.drop_duplicates("place_id")
        m = match_places(pi, ds)
        m.merge(pi[["place_id", "name", "naver_address"]], on="place_id") \
         .merge(ds[["restaurant_id", "사업장명", "도로명주소", "event"]], on="restaurant_id") \
         .to_csv("review_match_table.csv", index=False, encoding="utf-8-sig")

    pi["review_volume"] = np.log1p(pd.to_numeric(pi["visitor_review_total"], errors="coerce"))
    summ, rv_tagged = summarize_reviews(rv)
    feats = pi[["place_id", "review_volume"]].drop_duplicates("place_id").merge(summ, on="place_id", how="left")

    mcols = [c for c in ["restaurant_id", "place_id", "match_status", "shared_place"] if c in m.columns]
    out = ds.merge(m[mcols], on="restaurant_id", how="left") \
            .merge(feats, on="place_id", how="left")
    out["has_review"] = out["place_id"].notna().astype(int)
    out.to_csv("final_dataset_with_reviews.csv", index=False, encoding="utf-8-sig")

    h = out[out.has_review == 1]
    print(f"네이버 매장 {len(pi):,} | 매칭 {len(h):,}/{len(out):,} ({len(h) / len(out):.1%})")
    print("\n[event x has_review]  폐업(1) 쪽 has_review=1 이 거의 없으면 has_review를 Cox에 넣지 말 것")
    print(pd.crosstab(out["event"], out["has_review"], margins=True))
    print("\n[매칭 매장 내 수집률]")
    print(h[["review_volume", "review_segment", "review_revisit_share"]].notna().mean().round(3).to_string())
    print("\n[review_segment 분포]")
    print(h["review_segment"].value_counts(dropna=False).to_string())
    print(f"\n[리뷰 단위] 손님층 태그가 잡힌 리뷰 {rv_tagged['segment'].notna().mean():.1%}")
    print(rv_tagged["segment"].value_counts(normalize=True).round(3).to_string())

    # 실제 태그 확인용
    ch = rv["meta_text"].map(tag_chunk)
    print(f"\n[태그 조합 상위 25]  태그가 있는 리뷰 {ch.notna().mean():.1%}  (목적 / 동행자)")
    print(", ".join(f"{k}({v})" for k, v in ch.value_counts().head(25).items()))
    # 기준점 뒤에 나온 단어 중 선택지에 없는 것 -> 새 태그일 수 있음
    def head_word(meta):
        t = re.sub(r"[・•ㆍ･]", "·", str(meta)).split("반응 남기기")[0]
        ms = list(re.finditer(ANCHOR, t))
        return t[ms[-1].end():].lstrip(" |").split(" ")[0][:12] if ms else None
    hw = rv.loc[ch.isna(), "meta_text"].map(head_word).dropna()
    hw = hw[~hw.str.endswith("요") & (hw.str.len() > 0)]
    print("\n[선택지에 없는 단어 후보] 손님층 태그면 VOCAB에 추가:",
          ", ".join(f"{k}({v})" for k, v in hw.value_counts().head(15).items()) or "없음")

if __name__ == "__main__":
    main(*sys.argv[1:4])
