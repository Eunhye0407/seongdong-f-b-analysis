"""
not_found 원인 진단: 지정한 점포의 검색 과정을 단계별로 기록 (화면 캡처 + 텍스트)

사용법:  python3 diagnose_search.py
결과  :  diagnose/ 폴더  -> 이 폴더를 통째로(또는 summary.txt + png) 보내주세요
"""
import asyncio
import os
import re
from urllib.parse import quote

import pandas as pd
from playwright.async_api import async_playwright
from naver_crawling_cohort import make_query, current_place_id

DATASET_CSV = "final_restaurant_level_dataset.csv"      # 도로명주소가 있는 예전 파일(3,779행)
TARGET_IDS = ["3030000-101-2020-00521",                  # 연남토마 성수점
              "3030000-101-2017-00409"]                  # 어라운드데이
OUT = "diagnose"
HEADLESS = False                                         # False: 브라우저 창이 떠서 직접 볼 수 있음


async def main():
    os.makedirs(OUT, exist_ok=True)
    ds = pd.read_csv(DATASET_CSV, encoding="utf-8-sig")
    rows = ds[ds["restaurant_id"].isin(TARGET_IDS)]
    log = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        page = await (await browser.new_context(locale="ko-KR", viewport={"width": 1280, "height": 900})).new_page()
        for _, r in rows.iterrows():
            for qi, q in enumerate([x for x in make_query(r["사업장명"], r["도로명주소"], r.get("지번주소")) if x]):
                tag = f"{r['restaurant_id']}_q{qi + 1}"
                await page.goto(f"https://map.naver.com/p/search/{quote(q)}", wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(8000)
                has_search = await page.locator("iframe#searchIframe").count() > 0
                has_entry = await page.locator("iframe#entryIframe").count() > 0
                pid = await current_place_id(page)
                src = await page.locator("iframe#entryIframe").first.get_attribute("src") if has_entry else None
                info = [f"[{tag}] {r['사업장명']}", f"  검색어      : {q}", f"  최종 URL    : {page.url}",
                        f"  place id    : {pid or '없음'}",
                        f"  상세 iframe src : {src}",
                        f"  목록 iframe : {has_search} | 상세 iframe : {has_entry}"]
                if has_search:
                    fr = await (await page.query_selector("iframe#searchIframe")).content_frame()
                    txt = await fr.locator("body").inner_text()
                    for sel in ["li:has(span.TYaxT)", "li:has(span.YwYLL)", "li:has(span.place_bluelink)", "li"]:
                        info.append(f"  {sel:28s}: {await fr.locator(sel).count()}개")
                    html = await fr.content()
                    open(f"{OUT}/{tag}_list.txt", "w", encoding="utf-8").write(txt)
                    open(f"{OUT}/{tag}_list.html", "w", encoding="utf-8").write(html)
                else:
                    open(f"{OUT}/{tag}_page.txt", "w", encoding="utf-8").write(await page.locator("body").inner_text())
                await page.screenshot(path=f"{OUT}/{tag}.png")
                print("\n".join(info)); log.extend(info + [""])
        await browser.close()
    open(f"{OUT}/summary.txt", "w", encoding="utf-8").write("\n".join(log))
    print(f"\n완료 -> {OUT}/summary.txt 와 png 파일들을 보내주세요")


if __name__ == "__main__":
    asyncio.run(main())
