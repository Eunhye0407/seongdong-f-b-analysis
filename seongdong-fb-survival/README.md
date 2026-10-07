# 성동구 F&B 점포 생존 분석: 경쟁 네트워크 · 네이버 리뷰 · System Dynamics

서울시 성동구 일반음식점의 폐업 위험을 경쟁 네트워크 구조와 입지 요인으로 설명하고(Cox 비례위험모형),
추정 결과를 System Dynamics 정책 시뮬레이션에 연결하는 연구의 코드와 데이터입니다.

## 연구 설계
- **기준 집단**: 2021년 1분기 말(2021-04-01) 영업 중인 성동구 일반음식점 **3,775개**
- **관찰 기간**: 2021-04-01 ~ 2026-09-12 (폐업 1,656개, 43.9%)
- **분석**: Kaplan-Meier → 블록별 Cox 비례위험모형 → System Dynamics

## 폴더 구성
```
code/
  naver_crawling_cohort.py   영업 점포를 한 곳씩 네이버 지도에서 검색 → 주소 검증 → 리뷰 수집
  make_hints.py              동 키워드 수집 결과로 미매칭 점포의 place_id 힌트 생성
  combine_parts.py           나눠서 수집한 part 파일 합치기
  review_merge.py            리뷰 → 점포별 요약 변수 3개 → 분석 데이터셋에 결합
  diagnose_search.py         검색 실패 원인 진단용
  requirements.txt
notebooks/
  preprocessing_submission.ipynb   인허가·상권·매출·임대료·경쟁 변수 전처리
data/
  final_dataset_with_reviews.csv   분석용 최종 데이터 (점포 1행, 3,775행)
  naver_match_results.csv          인허가 점포 ↔ 네이버 매장 매칭 결과
```

## 데이터 출처
| 구분 | 자료 | 출처 |
|---|---|---|
| 생존(종속) | 성동구 일반음식점 인허가 정보 | 서울 열린데이터광장 |
| 상권·매출 | 상권분석서비스 상권영역·점포·추정매출 (2021Q1) | 서울 열린데이터광장 |
| 임대료 | 성동구 상가임대료 (2021Q1) | 성동구 |
| 좌표 보완 | Kakao Local API | Kakao |
| 리뷰 | 네이버 지도 플레이스 방문자 리뷰 | 직접 수집 (2026.10) |

## 주요 변수 (`final_dataset_with_reviews.csv`)
| 변수 | 설명 |
|---|---|
| `event`, `duration_days`, `duration_months` | 폐업 여부, 기준일부터 생존기간 |
| `business_age_at_baseline_years` | 기준일 시점 영업연령 |
| `TRDAR_CD`, `area_category_sales_per_store_final` | 상권 코드, 상권×업종 점포당 추정매출 |
| `rent_mean`, `rent_median` | 법정동별 임대료 |
| `restaurants_all_300m`, `competitors_same_300m` | 반경 300m 내 음식점 수, 동일업종 경쟁점포 수 |
| `review_volume` | log(1 + 방문자 리뷰 총수): 수요 규모 |
| `review_segment` | 성동구 평균 대비 특히 많은 손님층 (연인·가족·친구·업무·혼자) |
| `review_revisit_share` | 재방문 리뷰 비율: 고객 충성도 |
| `has_review`, `match_status`, `shared_place` | 리뷰 매칭 여부·상태, 한 매장에 인허가 여러 건 여부 |

## 리뷰 수집 요약
- 대상: 관찰 종료일까지 영업 중인 2,119개 점포
- 방법: "상호명 + 도로명 건물번호" 검색 → 네이버 매장 도로명주소가 인허가 주소와 **같은 건물일 때만** 확정
- 결과: **1,395개 점포(65.8%) 매칭, 리뷰 34,736건**. 검색 결과 자체가 없는 점포 220개(10.4%)는 실질 폐업 추정
- 리뷰 변수는 영업 점포에만 존재하므로, 기본 Cox 모형이 아닌 영업 점포 대상 확장 분석에 사용

## 실행
```bash
pip install -r code/requirements.txt
python -m playwright install chromium
python code/naver_crawling_cohort.py
python code/review_merge.py cohort_part0_reviews.csv cohort_part0_restaurants.csv final_restaurant_level_dataset.csv
```

## 개인정보
`data/`에는 점포 단위 요약 변수만 포함됩니다.
