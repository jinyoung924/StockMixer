"""
CSI300 데이터셋 빌드 — StockMixer `(N,T,F)` pkl 계약 (data build.md §6 파이프라인).

산출(dataset/CSI300/, float32, 행=종목 N, 열=거래일 T, 순서·mask/gt/price 전 arm 동일):
  eod_data_5.pkl   (N,T,5)   baseline 입력: [MA5,MA10,MA20,MA30,close]/norm (마지막=close)
  eod_data_13.pkl  (N,T,13)  add/concat/gating 입력: alpha158 KBAR9+price4 13피처
  price_data.pkl   (N,T)     종목별 정규화 close 레벨 (base_price)
  gt_data.pkl      (N,T)     forward return = (close[t]-close[t-steps])/close[t-steps]
  mask_data.pkl    (N,T)     1=close존재 ∧ 13피처존재 ∧ 라벨존재 ∧ 당일 CSI300 편입
  m_tau.pkl        (T,63)    MASTER csi_market_information 를 date_order 로 reindex
  meta.pkl         stock_order, date_order, valid_index, test_index, steps

핵심 원칙(data build.md §6.5):
  - m_τ 제외 모든 피처는 같은 qlib 소스 한 곳에서 생성(인덱스 구조적 일치).
  - 단일 유효성 마스크를 전 arm 동일 적용. price_data 는 per-종목 가격 레벨(횡단면 z 금지).
  - 정규화 fit(13피처 median/MAD, 5피처 norm)은 train 구간에서만.

전제: qlib cn_data(opensource) 설치 + MASTER/data/csi_market_information.csv.

사용 예:
  python build_dataset.py \
      --qlib-dir ~/.qlib/qlib_data/cn_data \
      --market-csv ../../../MASTER/data/csi_market_information.csv \
      --out ../../dataset/CSI300 \
      --start 2008-01-01 --end 2022-12-30 \
      --train-end 2018-12-31 --valid-end 2019-12-31 --steps 1
"""
import os
import pickle
import argparse
import numpy as np
import pandas as pd

# alpha158 KBAR(9) + price0(4) = 13피처. 컬럼명 드리프트를 피해 qlib 표현식으로 직접 정의.
ALPHA13 = {
    "KMID":  "($close-$open)/$open",
    "KLEN":  "($high-$low)/$open",
    "KMID2": "($close-$open)/($high-$low+1e-12)",
    "KUP":   "($high-Greater($open,$close))/$open",
    "KUP2":  "($high-Greater($open,$close))/($high-$low+1e-12)",
    "KLOW":  "(Less($open,$close)-$low)/$open",
    "KLOW2": "(Less($open,$close)-$low)/($high-$low+1e-12)",
    "KSFT":  "(2*$close-$high-$low)/$open",
    "KSFT2": "(2*$close-$high-$low)/($high-$low+1e-12)",
    "OPEN0": "$open/$close",
    "HIGH0": "$high/$close",
    "LOW0":  "$low/$close",
    "VWAP0": "$vwap/$close",
}
ALPHA13_NAMES = list(ALPHA13)


def init_qlib(qlib_dir):
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=os.path.expanduser(qlib_dir), region=REG_CN)


def build_grid(start, end, universe="csi300"):
    """격자 정의: stock_order(CSI300 구성종목 합집합), date_order(거래 캘린더)."""
    from qlib.data import D
    cal = [str(d.date()) for d in D.calendar(start_time=start, end_time=end)]
    insts = D.list_instruments(D.instruments(universe), start_time=start,
                               end_time=end, as_list=True)
    stock_order = sorted(insts)
    date_order = cal
    # 동적 유니버스 편입/편출 spell (instrument -> [(in,out),...])
    spells = D.list_instruments(D.instruments(universe), start_time=start,
                                end_time=end, as_list=False)
    return stock_order, date_order, spells


def _pivot(df_field, stock_order, date_order):
    """qlib (instrument,datetime) 시리즈 → (N,T) 격자. 없는 칸 NaN."""
    s = df_field.copy()
    s.index = s.index.set_levels(
        [s.index.levels[0], pd.to_datetime(s.index.levels[1])])
    wide = s.unstack(level=0)                      # index=datetime, columns=instrument
    wide.index = [str(d.date()) for d in wide.index]
    wide = wide.reindex(index=date_order, columns=stock_order)
    return wide.to_numpy(dtype=np.float64).T      # (N, T)


def fetch_fields(stock_order, date_order, start, end):
    """raw $close 와 alpha158 13피처를 한 번에 qlib 에서 가져와 (N,T) 격자로."""
    from qlib.data import D
    fields = ["$close"] + list(ALPHA13.values())
    names = ["close"] + ALPHA13_NAMES
    raw = D.features(stock_order, fields, start_time=start, end_time=end)
    raw.columns = names
    close = _pivot(raw["close"], stock_order, date_order)                 # (N,T)
    alpha = np.stack([_pivot(raw[n], stock_order, date_order)
                      for n in ALPHA13_NAMES], axis=-1)                    # (N,T,13)
    return close, alpha


def membership_mask(stock_order, date_order, spells):
    """당일 CSI300 편입이면 1. spell 구간으로 동적 유니버스 반영."""
    dates = pd.to_datetime(date_order)
    mask = np.zeros((len(stock_order), len(date_order)), dtype=np.float32)
    pos = {s: i for i, s in enumerate(stock_order)}
    for inst, periods in spells.items():
        if inst not in pos:
            continue
        i = pos[inst]
        for (s, e) in periods:
            lo, hi = pd.Timestamp(s), pd.Timestamp(e)
            mask[i, (dates >= lo) & (dates <= hi)] = 1.0
    return mask


def robust_zscore(alpha, train_index):
    """RobustZScoreNorm(train median/MAD fit) → ±3 clip → Fillna(0) (MASTER 규약).

    feature 별 전역(횡단면 아님) median/MAD 를 train 셀 전체로 fit.
    """
    out = alpha.copy().astype(np.float64)
    for c in range(alpha.shape[-1]):
        tr = alpha[:, :train_index, c]
        v = tr[np.isfinite(tr)]
        med = np.median(v) if v.size else 0.0
        mad = np.median(np.abs(v - med)) if v.size else 0.0
        scale = 1.4826 * mad if mad > 1e-12 else 1.0
        z = (out[:, :, c] - med) / scale
        out[:, :, c] = np.clip(z, -3.0, 3.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def build_close5(close, train_index):
    """종가 5피처 [MA5,MA10,MA20,MA30,close]/norm, norm=train 구간 종목별 close max."""
    N, T = close.shape
    cl = pd.DataFrame(close.T)            # (T,N) 행=시간 → rolling 용이
    ma = {w: cl.rolling(w, min_periods=1).mean().to_numpy().T for w in (5, 10, 20, 30)}
    norm = np.nanmax(close[:, :train_index], axis=1, keepdims=True)   # (N,1) 종목별 train max
    norm[~np.isfinite(norm) | (norm <= 0)] = 1.0
    feats = [ma[5], ma[10], ma[20], ma[30], close]
    eod5 = np.stack([f / norm for f in feats], axis=-1).astype(np.float32)   # (N,T,5)
    return eod5, (close / norm).astype(np.float32)    # price_data = close/norm = eod5[...,4]


def build_gt(price_norm, steps):
    """forward return 라벨 (per-종목 norm 상수에 불변 → 진짜 수익률)."""
    N, T = price_norm.shape
    gt = np.zeros((N, T), dtype=np.float32)
    gt[:, steps:] = (price_norm[:, steps:] - price_norm[:, :-steps]) / \
                    (price_norm[:, :-steps] + 1e-12)
    return gt


def load_m_tau(market_csv, date_order):
    """MASTER csi_market_information.csv (헤더 3줄: feature/expr/datetime) → (T,63)."""
    df = pd.read_csv(market_csv, skiprows=3, header=None, index_col=0)
    df.index = [str(pd.Timestamp(x).date()) for x in df.index]
    df = df.reindex(date_order).ffill().bfill()     # 거래일 정렬(index-파생, fit 불필요)
    m = df.to_numpy(dtype=np.float32)
    assert m.shape[1] == 63, f"m_τ 차원={m.shape[1]} (63 기대)"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlib-dir", default="~/.qlib/qlib_data/cn_data")
    ap.add_argument("--market-csv", required=True,
                    help="MASTER/data/csi_market_information.csv 경로")
    ap.add_argument("--out", default="../../dataset/CSI300")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default="2022-12-30")
    ap.add_argument("--train-end", required=True, help="train 마지막 날짜(포함), 예 2018-12-31")
    ap.add_argument("--valid-end", required=True, help="valid 마지막 날짜(포함), 예 2019-12-31")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--universe", default="csi300")
    args = ap.parse_args()

    init_qlib(args.qlib_dir)
    os.makedirs(args.out, exist_ok=True)

    # Step 1 — 격자
    stock_order, date_order, spells = build_grid(args.start, args.end, args.universe)
    N, T = len(stock_order), len(date_order)
    valid_index = int(np.searchsorted(np.array(date_order), args.train_end, side="right"))
    test_index = int(np.searchsorted(np.array(date_order), args.valid_end, side="right"))
    print(f"grid N={N} T={T} | valid_index={valid_index} test_index={test_index}")

    # Step 2~4 — 피처/가격/라벨/마스크
    close, alpha = fetch_fields(stock_order, date_order, args.start, args.end)
    eod13 = robust_zscore(alpha, valid_index)                       # (N,T,13)
    eod5, price_data = build_close5(close, valid_index)             # (N,T,5), (N,T)
    gt_data = build_gt(price_data, args.steps)                      # (N,T)

    # Step 5 — m_τ
    m_tau = load_m_tau(args.market_csv, date_order)                 # (T,63)

    # 단일 유효성 마스크(§6.5): close ∧ 13피처 ∧ 라벨 ∧ 당일 유니버스, 전 arm 동일
    has_close = np.isfinite(close)
    has_alpha = np.isfinite(alpha).all(axis=-1)
    has_label = np.zeros((N, T), dtype=bool)
    has_label[:, args.steps:] = np.isfinite(close[:, args.steps:]) & np.isfinite(close[:, :-args.steps])
    in_univ = membership_mask(stock_order, date_order, spells) > 0.5
    mask_data = (has_close & has_alpha & has_label & in_univ).astype(np.float32)

    # 마스크 밖 셀은 0 으로 깨끗이 (격자 NaN 제거)
    for arr in (eod5, eod13):
        arr[~np.isfinite(arr)] = 0.0
    price_data = np.nan_to_num(price_data).astype(np.float32)
    gt_data = np.nan_to_num(gt_data).astype(np.float32)

    # ---- 산출물 저장 ----
    def dump(name, obj):
        with open(os.path.join(args.out, name), "wb") as f:
            pickle.dump(obj, f)

    dump("eod_data_5.pkl", eod5)
    dump("eod_data_13.pkl", eod13)
    dump("price_data.pkl", price_data)
    dump("gt_data.pkl", gt_data)
    dump("mask_data.pkl", mask_data)
    dump("m_tau.pkl", m_tau)
    dump("meta.pkl", dict(stock_order=stock_order, date_order=date_order,
                          valid_index=valid_index, test_index=test_index, steps=args.steps))

    # ---- 빌드 후 assert (data build.md §8/§9) ----
    assert np.allclose(eod5[..., 4], price_data, atol=1e-5), "eod5[...,4] != price_data"
    assert not np.isnan(mask_data).any(), "mask 에 NaN"
    assert m_tau.shape == (T, 63), f"m_τ shape {m_tau.shape}"
    print("=" * 60)
    print(f"  저장 완료: {args.out}")
    print(f"  eod5{eod5.shape} eod13{eod13.shape} m_tau{m_tau.shape} "
          f"mask 유효율={mask_data.mean():.3f}")
    print(f"  assert 통과: eod5[...,4]==price, mask no-NaN, m_τ=(T,63)")
    print("=" * 60)


if __name__ == "__main__":
    main()
