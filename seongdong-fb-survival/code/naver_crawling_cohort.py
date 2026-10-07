"""
분석 대상 점포를 한 곳씩 찾아가는 네이버 리뷰 크롤러

  입력 : final_restaurant_level_dataset.csv 의 영업 중 점포 (event=0, 2,119개)
  방법 : '상호명 + 성동구 도로명 건물번호'로 검색 -> 후보 매장의 도로명주소가 데이터셋 주소와
         같은 건물인지 확인 -> 맞으면 방문자 리뷰 총수 + 최신 리뷰 30개 수집
  결과 : cohort_partK_restaurants.csv  (restaurant_id 1행: place_id, 매칭 상태, 리뷰 총수 ...)
         cohort_partK_reviews.csv      (리뷰 1행: restaurant_id, place_id, visit_count, meta_text ...)

실행 :  python3 naver_crawling_cohort.py                 (전체를 혼자)
        python3 naver_crawling_cohort.py 0 3             (3명이 나눠서: 0번째 몫)
        python3 naver_crawling_cohort.py 1 3 / 2 3       (나머지 팀원)
"""
import asyncio
import hashlib
import os
import random
import re
import sys
from difflib import SequenceMatcher
from urllib.parse import quote

import pandas as pd
from playwright.async_api import async_playwright

# ------------------------------------------------------------
# 💡 실행 설정
# ------------------------------------------------------------
DATASET_CSV = "final_restaurant_level_dataset.csv"
PART = int(sys.argv[1]) if len(sys.argv) > 2 else 0          # 몇 번째 몫인지
N_PARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 1       # 몇 명이 나누는지
MAX_REVIEWS = 30
MAX_CANDIDATES = 3         # 검색 결과에서 확인할 후보 수
CONCURRENCY = 1            # 동시 탭 수. 2 이상이면 '과도한 접근' 차단이 잦음
BLOCK_WAIT_SEC = 90        # 차단 화면이 뜨면 기다렸다가 재시도하는 시간
BLOCK_RESOURCES = True
N_DEBUG = 3                # 앞 N개 점포는 검색 화면 원문 저장
RETRY_UNMATCHED = True     # True: 이전 실행의 not_found·addr_mismatch 점포를 새 로직으로 다시 시도
HINTS_CSV = "place_hints.csv"   # (restaurant_id, place_id) 목록이 있으면 검색 없이 그 매장부터 확인

OUT_REST = f"cohort_part{PART}_restaurants.csv"
OUT_REV = f"cohort_part{PART}_reviews.csv"
DEBUG_DIR = "debug_text"


# ============================================================
# 헬퍼
# ============================================================
def to_int(s):
    s = re.sub(r"[^\d]", "", s or "")
    return int(s) if s else None

def addr_key(a):
    """'서울특별시 성동구 연무장길 10, 1층 (성수동2가)' -> '연무장길10'"""
    m = re.search(r"([가-힣0-9]+(?:로|길))\s*(\d+(?:-\d+)?)", str(a))
    return f"{m.group(1)}{m.group(2)}" if m else None

def norm_name(x):
    x = re.sub(r"\(.*?\)|\[.*?\]|주식회사|\(주\)|㈜", "", str(x).lower())
    return re.sub(r"[^0-9a-z가-힣]", "", x)

def name_sim(a, b):
    a, b = norm_name(a), norm_name(b)
    if not a or not b: return 0.0
    if a == b: return 1.0
    if a in b or b in a: return 0.9
    return SequenceMatcher(None, a, b).ratio()

def make_query(name, road_addr, jibun_addr=None):
    clean = re.sub(r"\(.*?\)|\[.*?\]|주식회사|\(주\)|㈜", " ", str(name)).strip()
    m = re.search(r"([가-힣0-9]+(?:로|길))\s*(\d+(?:-\d+)?)", str(road_addr))
    dong = re.search(r"성동구\s+([가-힣0-9]+동[0-9]*가?)", str(jibun_addr))
    return [f"{clean} 성동구 {m.group(1)} {m.group(2)}" if m else None,   # 1차: 상호명 + 도로명 건물번호
            f"{clean} 성동구 {dong.group(1)}" if dong else None,           # 2차: 상호명 + 법정동
            f"{clean} 성동구"]                                            # 3차: 상호명 + 성동구

def make_review_id(place_id, text, date):
    return hashlib.sha256(f"{place_id}|{date}|{text}".encode("utf-8")).hexdigest()

def extract_full_date(t):
    m = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*[월화수목금토일]요일", t)
    return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None

def extract_visit_count(t):
    m = re.search(r"(\d+)\s*번째 방문", t)
    return int(m.group(1)) if m else None

def parse_home(text, html):
    visitor = None
    for p in [r"방문자\s*리뷰\s*([\d,]+)", r"리뷰\s*([\d,]+)\s*·\s*블로그"]:
        m = re.search(p, text)
        if m: visitor = to_int(m.group(1)); break
    if visitor is None:
        m = re.search(r'"visitorReviewsTotal"\s*:\s*"?(\d+)', html)
        if m: visitor = int(m.group(1))
    addr = re.search(r"(서울\S*\s+성동구\s+[가-힣0-9]+(?:로|길)\s*\d+(?:-\d+)?)", text)
    addr = addr.group(1) if addr else None
    if addr is None:                                                # 화면에 없으면 페이지 소스(JSON)에서
        m = re.search(r'"roadAddress"\s*:\s*"([^"]+)"', html) or re.search(r'"(서울[^"]{0,10}성동구[^"]+)"', html)
        addr = m.group(1) if m else None
    name = None
    for p in [r'property="og:title"\s+content="([^"]+)"', r'"name"\s*:\s*"([^"]{1,40})"', r"<title>(.*?)\s*[-:]\s*네이버"]:
        m = re.search(p, html)
        if m: name = m.group(1).strip(); break
    return {"visitor_review_total": visitor, "naver_address": addr, "naver_name": name}

def road_name(a):
    m = re.search(r"([가-힣0-9]+(?:로|길))\s*\d", str(a))
    return m.group(1) if m else None

def save_debug(name, text):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    with open(os.path.join(DEBUG_DIR, name), "w", encoding="utf-8") as f:
        f.write(text)

async def block_heavy(route):
    if route.request.resource_type in ("image", "media", "font"):
        await route.abort()
    else:
        await route.continue_()


# ============================================================
# 1) 검색 -> 후보 place_id 목록
# ============================================================
BLOCK_MSG = "과도한 접근"
NO_RESULT_MSG = ("조건에 맞는 업체가 없습니다", "검색결과가 없습니다", "검색 결과가 없습니다")

# 검색 목록의 가게 항목을 클래스명 없이 찾기: '상세주소'·'출발'이 들어간 최상위 li = 가게 1개
ITEMS_JS = """() => {
  const lis = [...document.querySelectorAll('li')].filter(li =>
      !li.parentElement.closest('li') && /상세주소|출발/.test(li.innerText));
  return lis.map((li, i) => {
    li.setAttribute('data-cc', i);
    const lines = li.innerText.split('\\n').map(s => s.trim()).filter(Boolean);
    const skip = /^(출발|도착|예약|톡톡|상세주소|네이버페이|배달|주문|플레이스|쿠폰|새로오픈)/;
    const a = [...li.querySelectorAll('a')].find(a => a.innerText.trim() && !skip.test(a.innerText.trim()));
    if (a) a.setAttribute('data-cc-a', i);
    return {i, name: lines[0] || '', addr: lines.find(l => /^서울\\s/.test(l)) || '', hasA: !!a};
  });
}"""


async def current_place_id(page):
    """지금 열린 매장 상세의 place id. 주소창(/place/123)에 없으면 상세 iframe 주소에서 읽음
    (요즘 네이버는 검색 결과가 하나면 주소창에 id 없이 'placePath=/home...'만 붙이고 상세를 바로 엶)"""
    m = re.search(r"/place/(\d+)", page.url)
    if m: return m.group(1)
    ent = page.locator("iframe#entryIframe")
    if await ent.count() == 0: return None
    src = await ent.first.get_attribute("src") or ""
    m = re.search(r"place\.naver\.com/[a-z]+/(\d+)", src) or re.search(r"/(\d{6,})(?:/|\?|$)", src)
    if m: return m.group(1)
    try:
        fr = await (await page.query_selector("iframe#entryIframe")).content_frame()
        m = re.search(r"place\.naver\.com/[a-z]+/(\d+)", fr.url if fr else "")
        return m.group(1) if m else None
    except Exception:
        return None


async def _search_once(page, query, target_name, dong=None, debug_name=None):
    """-> (후보 [(place_id, 목록 이름)], 상태 'ok' | 'no_result' | 'blocked')"""
    await page.goto(f"https://map.naver.com/p/search/{quote(query)}", wait_until="domcontentloaded", timeout=30000)
    for _ in range(40):                                            # 최대 약 8초: 상세 or 목록 대기
        await page.wait_for_timeout(200)
        pid = await current_place_id(page)                         # 결과 1개 -> 상세가 바로 열림
        if pid: return [(pid, None)], "ok"
        if await page.locator("iframe#searchIframe").count() > 0: break
    body = await page.locator("body").inner_text()
    if BLOCK_MSG in body: return [], "blocked"
    handle = await page.query_selector("iframe#searchIframe")
    frame = await handle.content_frame() if handle else None
    if frame is None: return [], "no_result"
    try:
        await frame.wait_for_function("() => /상세주소|출발|조건에 맞는|과도한 접근/.test(document.body.innerText)", timeout=6000)
    except Exception:
        pass
    text = await frame.locator("body").inner_text()
    if debug_name: save_debug(f"{debug_name}_search.txt", f"[검색어] {query}\n\n{text}")
    if BLOCK_MSG in text: return [], "blocked"
    if any(msg in text for msg in NO_RESULT_MSG): return [], "no_result"

    items = await frame.evaluate(ITEMS_JS)
    scored = []
    for it in items[:15]:
        if "성동구" not in it["addr"] and it["addr"]: continue        # 다른 구의 가게는 제외
        sim = name_sim(target_name, it["name"])                    # '외가집육류,고기요리'처럼 업종이 붙어도 포함관계로 0.9
        same_dong = bool(dong) and dong in it["addr"]
        if sim >= 0.5 or (same_dong and sim >= 0.35):              # 같은 동이면 오타('설농탕'/'설렁탕')도 후보로
            scored.append((sim + 0.05 * same_dong, it))
    out = []
    for _, it in sorted(scored, key=lambda x: -x[0])[:MAX_CANDIDATES]:
        old_id = await current_place_id(page)
        target = frame.locator(f"[data-cc-a='{it['i']}']") if it["hasA"] else frame.locator(f"li[data-cc='{it['i']}']")
        try:
            await target.first.click(timeout=4000)
        except Exception:
            continue
        for _ in range(25):
            await page.wait_for_timeout(200)
            pid = await current_place_id(page)
            if pid and pid != old_id:
                out.append((pid, it["name"])); break
    return out, "ok"


async def search_candidates(page, query, target_name, dong=None, debug_name=None):
    """화면 전환 오류는 3번, 차단 화면은 BLOCK_WAIT_SEC 기다렸다가 2번까지 재시도"""
    status = "ok"
    for attempt in range(4):
        try:
            cands, status = await _search_once(page, query, target_name, dong, debug_name)
        except Exception:
            await page.wait_for_timeout(1000 * (attempt + 1)); continue
        if status != "blocked": return cands, status
        print(f"  ⏸ 네이버 접근 제한 -> {BLOCK_WAIT_SEC}초 대기 후 재시도")
        await page.wait_for_timeout(BLOCK_WAIT_SEC * 1000 * (attempt + 1))
    return [], status


# ============================================================
# 2) 매장 홈 (주소 검증, 리뷰 총수)
# ============================================================
async def crawl_home(page, place_id):
    try:
        await page.goto(f"https://pcmap.place.naver.com/restaurant/{place_id}/home",
                        wait_until="domcontentloaded", timeout=30000)
        try:
            await page.wait_for_function("document.body.innerText.includes('리뷰')", timeout=5000)
        except Exception:
            pass
        return parse_home(await page.locator("body").inner_text(), await page.content())
    except Exception:
        return {}


# ============================================================
# 3) 리뷰 탭 - 본문 있는 최신 리뷰 target개
# ============================================================
VALID_COUNT_JS = """() => [...document.querySelectorAll('ul#_review_list > li.EjjAW')]
  .filter(li => { const a = li.querySelector('div.pui__vn15t2 > a'); return a && a.innerText.trim(); }).length"""
EXTRACT_JS = """() => [...document.querySelectorAll('ul#_review_list > li.EjjAW')].map(li => {
  const a = li.querySelector('div.pui__vn15t2 > a');
  return {full: li.innerText || '', body: a ? a.innerText.trim() : ''};
})"""

async def sort_by_latest(page):
    btns = page.locator("a.place_btn_option")
    for i in range(await btns.count()):
        try:
            if "최신순" in await btns.nth(i).inner_text():
                await btns.nth(i).click(); await page.wait_for_timeout(800); return
        except Exception:
            continue

async def load_until(page, target):
    no_growth, waits = 0, [500, 800, 1200, 2000, 3000, 4500]
    for _ in range(80):
        if await page.evaluate(VALID_COUNT_JS) >= target: return
        before = await page.locator("ul#_review_list > li.EjjAW").count()
        clicked = False
        for sel in ["div.NSTUp a.fvwqf", "a.fvwqf"]:
            btn = page.locator(sel)
            if await btn.count() > 0:
                try:
                    await btn.first.click(timeout=3000); clicked = True; break
                except Exception:
                    pass
        if not clicked:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        try:
            await page.wait_for_function(
                "n => document.querySelectorAll('ul#_review_list > li.EjjAW').length > n",
                arg=before, timeout=waits[min(no_growth, len(waits) - 1)])
            no_growth = 0
        except Exception:
            no_growth += 1
            if no_growth >= len(waits): return

async def crawl_reviews(page, rid, place_id, target):
    best = []
    for _ in range(2):
        try:
            await page.goto(f"https://pcmap.place.naver.com/restaurant/{place_id}/review/visitor",
                            wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_selector("ul#_review_list", timeout=12000)
        except Exception:
            continue
        await sort_by_latest(page)
        await load_until(page, target)
        rows, seen = [], set()
        for it in await page.evaluate(EXTRACT_JS):
            if len(rows) >= target: break
            body, full = it["body"], it["full"]
            if not body: continue
            date = extract_full_date(full)
            rv_id = make_review_id(place_id, body, date)
            if rv_id in seen: continue
            seen.add(rv_id)
            lines = [l.strip() for l in full.replace(body, "\n").split("\n") if l.strip()]
            rows.append({"review_id": rv_id, "restaurant_id": rid, "place_id": place_id,
                         "review_date": date, "visit_count": extract_visit_count(full),
                         "meta_text": " | ".join(lines)[:500], "review_text": body})
        if len(rows) > len(best): best = rows
        if len(best) >= target: break
    return best


# ============================================================
# 4) 점포 1곳 처리: 검색 -> 주소 검증 -> 리뷰
# ============================================================
async def process_store(page, row, debug=False):
    rid, name, road, jibun = row["restaurant_id"], row["사업장명"], row["도로명주소"], row.get("지번주소")
    want, want_road = addr_key(road), road_name(road)
    res = {"restaurant_id": rid, "사업장명": name, "도로명주소": road, "place_id": None,
           "naver_name": None, "naver_address": None, "visitor_review_total": None,
           "match_status": "not_found", "review_n": 0}
    tried, fallback = set(), None
    queries = [q for q in make_query(name, road, jibun) if q]
    dm = re.search(r"성동구\s+([가-힣0-9]+동[0-9]*가?)", str(jibun))
    dong = dm.group(1) if dm else None
    statuses = []
    if rid in HINTS:                                                   # 이미 알고 있는 place_id -> 검색 생략
        queries = ["__HINT__"] + queries
    for qi, q in enumerate(queries):
        last = qi == len(queries) - 1
        if q == "__HINT__":
            cands = [(HINTS[rid], None)]
        else:
            cands, st = await search_candidates(page, q, name, dong, debug_name=f"{rid}" if (debug or last) else None)
            statuses.append(st)
        for pid, list_nm in cands:
            if pid in tried: continue
            tried.add(pid)
            home = await crawl_home(page, pid)
            nv_name = home.get("naver_name") or list_nm or ""
            got = addr_key(home.get("naver_address"))
            sim = name_sim(name, nv_name) if nv_name else 0
            if want and got == want:                                   # ① 같은 건물
                res.update(home, place_id=pid, match_status="matched"); break
            if not want and sim >= 0.8:                                # ② 도로명주소 없는 점포: 이름
                res.update(home, place_id=pid, match_status="matched_name"); break
            if want_road and road_name(home.get("naver_address")) == want_road and sim >= 0.8:
                fallback = fallback or ("matched_road", pid, home)     # ③ 같은 도로 + 이름 매우 유사 (건물번호만 다름)
            elif got is None and sim >= 0.9:
                fallback = fallback or ("matched_name", pid, home)     # ④ 주소를 못 읽었지만 이름이 거의 같음
            elif res["match_status"] == "not_found":
                res.update(match_status="addr_mismatch", naver_address=home.get("naver_address"), naver_name=nv_name)
        if res["match_status"].startswith("matched"): break
    if not res["match_status"].startswith("matched") and fallback:
        st, pid, home = fallback
        res.update(home, place_id=pid, match_status=st)
    if res["match_status"] == "not_found" and statuses:
        if "blocked" in statuses: res["match_status"] = "blocked"              # 차단 때문에 못 찾음 -> 다음에 재시도
        elif all(x == "no_result" for x in statuses): res["match_status"] = "no_result"   # 네이버에 검색 결과 자체가 없음

    reviews = []
    if res["match_status"].startswith("matched"):
        total = res["visitor_review_total"]
        target = MAX_REVIEWS if total is None else int(min(MAX_REVIEWS, total))
        if target > 0:
            reviews = await crawl_reviews(page, rid, res["place_id"], target)
        res["review_n"] = len(reviews)
    return res, reviews


# ============================================================
# Main
# ============================================================
HINTS = {}

async def main():
    global HINTS
    if os.path.exists(HINTS_CSV):
        h = pd.read_csv(HINTS_CSV, encoding="utf-8-sig", dtype={"place_id": str})
        HINTS = dict(zip(h["restaurant_id"], h["place_id"]))
        print(f"place_hints: {len(HINTS)}개 점포는 검색 없이 바로 확인")
    ds = pd.read_csv(DATASET_CSV, encoding="utf-8-sig")
    ds = ds[ds["event"] == 0].reset_index(drop=True)
    ds = ds[ds.index % N_PARTS == PART].reset_index(drop=True)

    done, all_reviews = pd.DataFrame(), []
    if os.path.exists(OUT_REST):
        done = pd.read_csv(OUT_REST, encoding="utf-8-sig", dtype={"place_id": str})
    if os.path.exists(OUT_REV):
        all_reviews = pd.read_csv(OUT_REV, encoding="utf-8-sig", dtype={"place_id": str}).to_dict("records")
    if len(done):                                                   # 오류 났던 점포는 다시 시도
        retry = done["match_status"].astype(str).str.startswith("error")
        if RETRY_UNMATCHED:
            retry |= done["match_status"].isin(["not_found", "addr_mismatch", "rejected_other_branch", "blocked"])
        done = done[~retry]
    done_ids = set(done["restaurant_id"]) if len(done) else set()
    results = done.to_dict("records") if len(done) else []
    todo = ds[~ds["restaurant_id"].isin(done_ids)]
    print("=" * 70)
    print(f"part {PART}/{N_PARTS} | 대상 {len(ds)}개 | 완료 {len(done_ids)}개 | 남은 {len(todo)}개 | 동시 탭 {CONCURRENCY}")
    print("=" * 70)

    queue = asyncio.Queue()
    for i, r in todo.iterrows(): queue.put_nowait((i, r))
    cnt = {"n": 0, "matched": 0}

    def save():
        pd.DataFrame(results).to_csv(OUT_REST, index=False, encoding="utf-8-sig")
        if all_reviews:
            pd.DataFrame(all_reviews).drop_duplicates("review_id").to_csv(OUT_REV, index=False, encoding="utf-8-sig")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = await browser.new_context(
            locale="ko-KR", timezone_id="Asia/Seoul", viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
        if BLOCK_RESOURCES:
            await context.route("**/*", block_heavy)

        async def worker():
            page = await context.new_page()
            while not queue.empty():
                i, row = await queue.get()
                try:
                    res, revs = await process_store(page, row, debug=i < N_DEBUG)
                except Exception as e:
                    res, revs = {"restaurant_id": row["restaurant_id"], "사업장명": row["사업장명"],
                                 "match_status": f"error: {str(e)[:60]}"}, []
                results.append(res); all_reviews.extend(revs)
                cnt["n"] += 1; cnt["matched"] += str(res["match_status"]).startswith("matched")
                print(f"[{cnt['n']}/{len(todo)}] {row['사업장명']} | {res['match_status']}"
                      + (f" | 리뷰 {res['review_n']}/{min(MAX_REVIEWS, res['visitor_review_total'] or MAX_REVIEWS)}"
                         if str(res["match_status"]).startswith("matched") else "")
                      + f"  (누적 매칭률 {cnt['matched'] / cnt['n']:.0%})")
                save()
                await page.wait_for_timeout(random.randint(800, 2000))      # 연속 요청 간격 (차단 방지)
            await page.close()

        await asyncio.gather(*[worker() for _ in range(CONCURRENCY)])
        await browser.close()

    save()
    out = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print(f"part {PART} 완료 | 점포 {len(out)}개 | 리뷰 {len(all_reviews)}개")
    print(out["match_status"].value_counts().to_string())
    print("  · no_result     : 네이버 검색 결과 자체가 없음 -> 실제 폐업·상호 변경 가능성 높음")
    print("  · not_found     : 검색 결과는 있었지만 맞는 가게가 없음")
    print("  · blocked       : 네이버 접근 제한으로 실패 -> 다시 실행하면 재시도")
    print("  · addr_mismatch : 비슷한 이름은 있으나 다른 건물 -> 검수 대상")
    print("  · matched_road  : 같은 도로 + 이름 거의 같음 (건물번호만 다름) -> 검수 권장")
    print("  · matched_name  : 주소 없이 이름으로 확정 -> 검수 권장")
    print("  · error         : 같은 명령으로 다시 실행하면 자동 재시도")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
