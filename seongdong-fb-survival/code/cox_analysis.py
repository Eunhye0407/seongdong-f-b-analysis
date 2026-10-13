import pandas as pd, numpy as np, json
from pathlib import Path
from statsmodels.duration.hazard_regression import PHReg

# 경로: 이 파일(code/) 기준으로 프로젝트 루트를 찾음 → 어디서 실행해도 동작
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'final_dataset_with_reviews.csv'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# ---- 1. 경쟁 변수: 기준시점(2021-04-01) 영업 중인 이웃(=표본 전체) 기준, EPSG:5186 투영좌표 거리
from pyproj import Transformer
_t = Transformer.from_crs('EPSG:4326', 'EPSG:5186', always_xy=True)
_x, _y = _t.transform(df.longitude.values, df.latitude.values)
D = np.hypot(_x[:, None]-_x[None, :], _y[:, None]-_y[None, :]); np.fill_diagonal(D, np.inf)
code = np.array(df['서울시_서비스_업종_코드'].fillna('미분류').astype(object))
same = code[:, None] == code[None, :]

RADII = [100, 200, 300, 500]
for r in RADII:
    df[f'BIZ_{r}'] = (D <= r).sum(1)
    df[f'C_{r}'] = ((D <= r) & same).sum(1)
    df[f'R_{r}'] = np.where(df[f'BIZ_{r}'] > 0, df[f'C_{r}'] / df[f'BIZ_{r}'].replace(0, np.nan), 0.0)

check = {'all300_match': float((df.BIZ_300 == df.restaurants_all_300m).mean()),
         'same300_match': float((df.C_300 == df.competitors_same_300m).mean())}

# ---- 2. 통제 변수
df['age'] = df.business_age_at_baseline_years
# 임대료: rent_median 하나만 사용 (rent_mean 은 사용하지 않음)
#  - 연속형: log(rent_median)
#  - 범주형: 동 단위 값의 자연 구간(1,364→2,064, 2,633→3,044 사이의 큰 간격)으로 3단계 구분
#    → 동 하나가 두 구간에 걸치지 않음
RENT_SPEC = 'continuous'   # 'continuous' 또는 'category' (본 모형/강건성 전환용)
df['log_rent'] = np.log(df.rent_median)
df['rent_level'] = pd.cut(df.rent_median, bins=[0, 1700, 2900, np.inf], labels=['저', '중', '고'])
rent_dummies = pd.get_dummies(df['rent_level'], prefix='임대료').astype(float).drop(columns='임대료_저')  # 기준: 저

sales = np.log(df.area_category_sales_per_store_final)
df['sales_missing'] = sales.isna().astype(int)
df['log_sales'] = sales.fillna(sales.median())
df['업종'] = df['서울시_서비스_업종_코드_명'].fillna('미분류')
dummies = pd.get_dummies(df['업종'], prefix='업종', drop_first=False).astype(float)
dummies = dummies.drop(columns='업종_한식음식점')  # 기준: 한식

def make_controls(rent_spec):
    base = ['age', 'log_sales', 'sales_missing']
    if rent_spec == 'continuous':
        return pd.concat([df[['age', 'log_rent', 'log_sales', 'sales_missing']], dummies], axis=1)
    return pd.concat([df[base], rent_dummies, dummies], axis=1)

X_ctrl = make_controls(RENT_SPEC)

# 단위: C·BIZ 는 10개 단위, R 은 10%p 단위
def run(cols, scale):
    X = X_ctrl.copy()
    for c in cols:
        X[c] = df[c] / scale[c]
    X = X[cols + list(X_ctrl.columns)]
    m = PHReg(df.duration_days, X, status=df.event, ties='efron').fit()
    ci = m.conf_int()
    out = pd.DataFrame({'HR': np.exp(m.params), 'lo': np.exp(ci[:, 0]), 'hi': np.exp(ci[:, 1]),
                        'p': m.pvalues}, index=X.columns)
    # concordance (Harrell C)
    lp = X.values @ m.params
    return out, m, lp

def harrell_c(t, e, lp):
    t = np.asarray(t); e = np.asarray(e); lp = np.asarray(lp)
    num = den = 0.0
    order = np.argsort(t)
    for i in np.where(e == 1)[0]:
        mask = t > t[i]
        den += mask.sum()
        num += (lp[i] > lp[mask]).sum() + 0.5*(lp[i] == lp[mask]).sum()
    return num/den

# 비례위험 검토: 경쟁 변수 × log(time) 상호작용은 PHReg에서 어렵기에 Schoenfeld 잔차-시간 상관으로 대체
def ph_check(m, X_cols):
    sr = m.schoenfeld_residuals
    ev = df.event.values == 1
    t = np.log(df.duration_days.values[ev])
    res = {}
    for j, c in enumerate(X_cols):
        s = sr[ev, j]
        ok = ~np.isnan(s)
        from scipy.stats import pearsonr
        rho, p = pearsonr(t[ok], s[ok])
        res[c] = (rho, p)
    return res

results = {}
rows = []
for r in RADII:
    sc = {f'C_{r}': 10, f'BIZ_{r}': 10, f'R_{r}': 0.1}
    for spec, cols in [('A', [f'C_{r}', f'BIZ_{r}']), ('B', [f'R_{r}', f'BIZ_{r}'])]:
        out, m, lp = run(cols, sc)
        cidx = harrell_c(df.duration_days, df.event, lp)
        ph = ph_check(m, cols + list(X_ctrl.columns))
        for v in cols:
            rows.append({'radius': r, 'model': spec, 'var': v.split('_')[0], 'HR': out.loc[v, 'HR'],
                         'lo': out.loc[v, 'lo'], 'hi': out.loc[v, 'hi'], 'p': out.loc[v, 'p'],
                         'ph_rho': ph[v][0], 'ph_p': ph[v][1], 'C_index': cidx, 'n': len(df),
                         'events': int(df.event.sum())})
        results[f'{r}_{spec}'] = out
        if r == 300 and spec == 'B':
            ctrl_table = out
            ctrl_ph = ph

res = pd.DataFrame(rows)
desc = df[[f'{p}_{r}' for r in RADII for p in ['C', 'BIZ', 'R']]].describe().T[['mean', '50%', 'std', 'min', 'max']]
corr = df[[f'C_{r}' for r in RADII] + [f'BIZ_{r}' for r in RADII]].corr()

pd.set_option('display.width', 200)
print(check)
print(desc.round(3))
print(res.round(4).to_string())
print(ctrl_table.round(4))
print({k: tuple(round(x, 3) for x in v) for k, v in ctrl_ph.items()})

res.to_csv(OUT / 'cox_competition_results.csv', index=False, encoding='utf-8-sig')
ctrl_table.to_csv(OUT / 'cox_full_300m_modelB.csv', encoding='utf-8-sig')
desc.to_csv(OUT / 'competition_desc.csv', encoding='utf-8-sig')
df[['restaurant_id', 'rent_level'] + [f'{p}_{r}' for r in RADII for p in ['C', 'BIZ', 'R']]].to_csv(
    OUT / 'competition_variables.csv', index=False, encoding='utf-8-sig')

json.dump({'check': check, 'results': res.to_dict('records'),
           'full300B': ctrl_table.reset_index().rename(columns={'index': 'var'}).to_dict('records'),
           'desc': desc.reset_index().rename(columns={'index': 'var'}).to_dict('records'),
           'ph300B': {k: list(v) for k, v in ctrl_ph.items()}},
          open(OUT / 'cox_out.json', 'w'), ensure_ascii=False, default=float)

# ---- 강건성: 임대료를 범주형(저/중/고)으로 바꾼 모형 (반경별 모형 B)
rob = []
for r in RADII:
    X_ctrl = make_controls('category')
    out, m, lp = run([f'R_{r}', f'BIZ_{r}'], {f'R_{r}': 0.1, f'BIZ_{r}': 10})
    for v in out.index:
        if v.startswith(('R_', 'BIZ_', '임대료')):
            rob.append({'radius': r, 'var': v, **out.loc[v].to_dict()})
rob = pd.DataFrame(rob)
print(rob.round(4).to_string())
print(df.groupby('rent_level', observed=True).agg(n=('event', 'size'), closed=('event', 'mean'),
      dongs=('rent_dong', lambda x: ', '.join(sorted(x.unique())))).to_string())
rob.to_csv(OUT / 'cox_robust_rent_category.csv', index=False, encoding='utf-8-sig')
