"""
CSI300 데이터셋 빌드 — StockMixer `(N,T,F)` pkl 계약 (data build.md §6 파이프라인).

산출(--out, float32, 행=종목 N, 열=거래일 T, 순서·mask/gt/price 전 arm 동일):
  eod_data_5.pkl     (N,T,5)   baseline 입력: [MA5,MA10,MA20,MA30,close]/cnorm (close=idx 4)
  eod_data_ohlcv.pkl (N,T,5)   add/concat/gating 입력: [O,H,L,C]/cnorm + V/vnorm (close=idx 3)
  price_data.pkl     (N,T)     종목별 정규화 close 레벨 (base_price = close/cnorm)
  gt_data.pkl        (N,T)     forward return = (close[t]-close[t-steps])/close[t-steps]
  mask_data.pkl      (N,T)     1=close∧OHLCV∧라벨∧price>0∧당일 CSI300 편입
  m_tau.pkl          (T,63)    MASTER csi_market_information 를 date_order 로 reindex
  meta.pkl           stock_order, date_order, valid_index, test_index, steps

핵심 원칙(data build.md §6.5):
  - m_τ 제외 모든 피처는 같은 qlib 소스 한 곳에서 생성(인덱스 구조적 일치).
  - baseline·OHLCV 모두 StockMixer load_data.py 방식 정규화: 가격은 종목별 train-max-close(cnorm),
    거래량은 종목별 train-max-volume(vnorm). 횡단면 z-score·RobustZScoreNorm 금지.
  - price_data 는 두 입력 파일의 close 채널과 동일(cnorm 공유) → StockMixer get_loss 의 base_price.

전제: 1) get_qlib_data.py 로 qlib cn_data 준비, 2) MASTER csi_market_information.csv.

사용 예(로컬):
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

RAW_FIELDS = ["$open", "$high", "$low", "$close", "$volume"]
RAW_NAMES = ["open", "high", "low", "close", "volume"]


def init_qlib(qlib_dir):
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=os.path.expanduser(qlib_dir), region=REG_CN)


def build_grid(start, end, universe="csi300"):
    """격자 정의: stock_order(CSI300 구성종목 합집합), date_order(거래 캘린더)."""
    from qlib.data import D
    cal = [str(pd.Timestamp(d).date()) for d in D.calendar(start_time=start, end_time=end)]
    insts = D.list_instruments(D.instruments(universe), start_time=start,
                               end_time=end, as_list=True)
    stock_order = sorted(insts)
    return stock_order, cal


def membership_mask(qlib_dir, universe, stock_order, date_order):
    """당일 유니버스 편입이면 1 — qlib instruments/{universe}.txt 를 직접 파싱.

    주의: qlib D.list_instruments(as_list=False) 의 spell 반환 형식은 버전마다 달라
    (dict 값이 tuple/list/np.datetime64 등) in_univ 이 통째로 비는 사고가 있었다.
    instruments 파일(inst<TAB>start<TAB>end, spell 당 1줄)을 직접 읽어 버전 독립적으로
    편입/편출(동적 유니버스)을 재현한다.
    """
    path = os.path.join(os.path.expanduser(qlib_dir), "instruments", f"{universe}.txt")
    dates = pd.to_datetime(date_order)
    mask = np.zeros((len(stock_order), len(date_order)), dtype=np.float32)
    pos = {s: i for i, s in enumerate(stock_order)}
    n_rows = 0
    with open(path) as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            inst, s, e = parts[0], parts[1], parts[2]
            if inst not in pos:
                continue
            lo, hi = pd.Timestamp(s), pd.Timestamp(e)
            mask[pos[inst], (dates >= lo) & (dates <= hi)] = 1.0
            n_rows += 1
    if n_rows == 0:
        raise RuntimeError(
            f"membership: {path} 에서 stock_order 와 매칭되는 편입 구간이 0개. "
            f"파일 경로/형식(instrument<TAB>start<TAB>end) 확인.")
    return mask


def _pivot(series, stock_order, date_order):
    """qlib (instrument,datetime) 시리즈 → (N,T) 격자. 없는 칸 NaN."""
    s = series.copy()
    s.index = s.index.set_names(["instrument", "datetime"])
    wide = s.unstack("instrument")                        # index=datetime, columns=instrument
    wide.index = [str(pd.Timestamp(d).date()) for d in wide.index]
    wide = wide.reindex(index=date_order, columns=stock_order)
    return wide.to_numpy(dtype=np.float64).T              # (N, T)


def fetch_fields(stock_order, date_order, start, end):
    """raw OHLCV 5필드를 qlib 에서 받아 {name: (N,T)} 격자로."""
    from qlib.data import D
    raw = D.features(stock_order, RAW_FIELDS, start_time=start, end_time=end)
    raw.columns = RAW_NAMES
    return {n: _pivot(raw[n], stock_order, date_order) for n in RAW_NAMES}


def per_stock_trainmax(arr, train_index):
    """종목별 train 구간 max (누수 방지). 비유효(≤0/NaN)는 1.0 으로."""
    norm = np.nanmax(arr[:, :train_index], axis=1, keepdims=True)   # (N,1)
    norm[~np.isfinite(norm) | (norm <= 0)] = 1.0
    return norm


def build_close5(close, cnorm):
    """baseline 종가 5피처 [MA5,MA10,MA20,MA30,close]/cnorm (close=idx 4)."""
    cl = pd.DataFrame(close.T)            # (T,N): 행=시간 → rolling 용이
    ma = {w: cl.rolling(w, min_periods=1).mean().to_numpy().T for w in (5, 10, 20, 30)}
    feats = [ma[5], ma[10], ma[20], ma[30], close]
    eod5 = np.stack([f / cnorm for f in feats], axis=-1).astype(np.float32)   # (N,T,5)
    price_norm = (close / cnorm).astype(np.float32)    # = eod5[...,4]
    return eod5, price_norm


def build_ohlcv(o, h, l, c, v, cnorm, vnorm):
    """OHLCV 5피처: 가격 4채널 ÷ cnorm, 거래량 ÷ vnorm. close=idx 3, volume=idx 4."""
    eod = np.stack([o / cnorm, h / cnorm, l / cnorm, c / cnorm, v / vnorm],
                   axis=-1).astype(np.float32)
    return eod                                          # (N,T,5)


def build_gt(price_norm, steps):
    """forward return 라벨 (per-종목 cnorm 상수에 불변 → 진짜 수익률)."""
    gt = np.zeros_like(price_norm, dtype=np.float32)
    gt[:, steps:] = (price_norm[:, steps:] - price_norm[:, :-steps]) / \
                    (price_norm[:, :-steps] + 1e-12)
    return gt


def finalize_arrays(close, feat_finite, eod5, eod_ohlcv, price_norm, gt_data, in_univ, steps):
    """단일 유효성 마스크 산출 + price placeholder(분모 0 방지).

    price_data 는 get_loss 의 분모(base_price)다. 0 이면 (pred-0)/0 = inf, inf*mask(0)=NaN.
    따라서 (1) 유효성에 price>0 포함(유효인데 price<=0 셀은 드롭·보고), (2) 마스크 밖 price 는
    placeholder 1.0. eod5[...,4] 와 eod_ohlcv[...,3] 은 price 와 동일성을 유지한다.

    feat_finite: OHLCV 5채널이 모두 유한한 (N,T) 불리언(종목 피처 존재).
    반환: (eod5, eod_ohlcv, price_data, gt_data, mask_data, n_drop_nonpos_price)
    """
    N, T = close.shape
    has_close = np.isfinite(close)
    has_label = np.zeros((N, T), dtype=bool)
    has_label[:, steps:] = np.isfinite(close[:, steps:]) & np.isfinite(close[:, :-steps])
    has_pos_price = np.isfinite(price_norm) & (price_norm > 0)

    base_valid = has_close & feat_finite & has_label & in_univ
    n_drop = int((base_valid & ~has_pos_price).sum())     # 검증: 유효였으나 price<=0 인 셀 수
    mask_data = (base_valid & has_pos_price).astype(np.float32)

    eod5 = eod5.copy(); eod_ohlcv = eod_ohlcv.copy()
    eod5[~np.isfinite(eod5)] = 0.0                        # 입력 결측 → 0 (마스크로 가려짐)
    eod_ohlcv[~np.isfinite(eod_ohlcv)] = 0.0
    invalid = mask_data < 0.5
    price_data = np.where(has_pos_price, price_norm, 1.0).astype(np.float32)
    price_data[invalid] = 1.0                            # 마스크 밖 base_price = placeholder 1.0
    eod5[..., 4] = price_data                            # baseline close 채널 == price
    eod_ohlcv[..., 3] = price_data                       # OHLCV close 채널 == price
    gt_data = gt_data.astype(np.float32).copy()
    gt_data[~np.isfinite(gt_data)] = 0.0                 # NaN/inf 라벨 제거
    gt_data[invalid] = 0.0
    return eod5, eod_ohlcv, price_data, gt_data, mask_data, n_drop


def load_m_tau(market_csv, date_order):
    """MASTER csi_market_information.csv (헤더 3행: feature/expr/datetime) → (T,63)."""
    df = pd.read_csv(os.path.expanduser(market_csv), skiprows=3, header=None, index_col=0)
    df.index = [str(pd.Timestamp(x).date()) for x in df.index]
    df = df[~df.index.duplicated(keep="first")]     # CSV 내 중복 날짜(16개) 정리
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
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Step 1 — 격자
    stock_order, date_order = build_grid(args.start, args.end, args.universe)
    N, T = len(stock_order), len(date_order)
    date_arr = np.array(date_order)
    valid_index = int(np.searchsorted(date_arr, args.train_end, side="right"))
    test_index = int(np.searchsorted(date_arr, args.valid_end, side="right"))
    assert 0 < valid_index < test_index < T, \
        f"분할 경계 이상: valid={valid_index} test={test_index} T={T} " \
        f"(train-end/valid-end 가 [{date_order[0]}, {date_order[-1]}] 안인지 확인)"
    print(f"grid N={N} T={T} | valid_index={valid_index} test_index={test_index}")

    # Step 2~4 — OHLCV fetch → 정규화 → 피처/가격/라벨
    f = fetch_fields(stock_order, date_order, args.start, args.end)
    o, h, l, c, v = (f["open"], f["high"], f["low"], f["close"], f["volume"])
    cnorm = per_stock_trainmax(c, valid_index)                     # 가격 정규화 상수
    vnorm = per_stock_trainmax(v, valid_index)                     # 거래량 정규화 상수
    eod5, price_data = build_close5(c, cnorm)                      # (N,T,5), (N,T)
    eod_ohlcv = build_ohlcv(o, h, l, c, v, cnorm, vnorm)          # (N,T,5)
    gt_data = build_gt(price_data, args.steps)                     # (N,T)

    # Step 5 — m_τ
    m_tau = load_m_tau(args.market_csv, date_order)                # (T,63)

    # 단일 유효성 마스크(§6.5) + price placeholder
    feat_finite = np.isfinite(np.stack([o, h, l, c, v], axis=-1)).all(axis=-1)
    in_univ = membership_mask(args.qlib_dir, args.universe, stock_order, date_order) > 0.5
    print(f"in_univ(편입) 유효율={in_univ.mean():.3f} | close 유효율={np.isfinite(c).mean():.3f} | "
          f"OHLCV 유효율={feat_finite.mean():.3f}")
    eod5, eod_ohlcv, price_data, gt_data, mask_data, n_drop = finalize_arrays(
        c, feat_finite, eod5, eod_ohlcv, price_data, gt_data, in_univ, args.steps)
    print(f"검증: 유효였으나 price<=0 로 드롭된 셀 = {n_drop} "
          f"(placeholder 1.0 적용 → base_price 0 제거)")

    # 빈 마스크 즉시 차단(in_univ 전부 0 → mask 전부 0 → train=0/평가 NaN 사고 방지)
    vpd = mask_data.sum(0)
    if mask_data.sum() == 0:
        raise RuntimeError("mask 가 전부 0 입니다(유효 셀 없음). 위 in_univ/close/OHLCV "
                           "유효율 로그로 어느 조건이 비었는지 확인하세요.")
    print(f"mask: 총유효 {int(mask_data.sum())}, 일평균 {vpd.mean():.1f}종목, "
          f"유효<2 인 날 {int((vpd < 2).sum())}일")

    # ---- 저장 ----
    def dump(name, obj):
        with open(os.path.join(out_dir, name), "wb") as fp:
            pickle.dump(obj, fp)

    dump("eod_data_5.pkl", eod5)
    dump("eod_data_ohlcv.pkl", eod_ohlcv)
    dump("price_data.pkl", price_data)
    dump("gt_data.pkl", gt_data)
    dump("mask_data.pkl", mask_data)
    dump("m_tau.pkl", m_tau)
    dump("meta.pkl", dict(stock_order=stock_order, date_order=date_order,
                          valid_index=valid_index, test_index=test_index, steps=args.steps))

    # ---- 빌드 후 assert (data build.md §8/§9) ----
    assert np.allclose(eod5[..., 4], price_data, atol=1e-5), "eod5[...,4] != price_data"
    assert np.allclose(eod_ohlcv[..., 3], price_data, atol=1e-5), "eod_ohlcv[...,3] != price_data"
    assert (price_data > 0).all(), "price_data 에 0/음수 존재 → base_price 분모 0 위험"
    assert np.isfinite(gt_data).all(), "gt_data 에 NaN/inf 존재"
    assert not np.isnan(mask_data).any(), "mask 에 NaN"
    assert m_tau.shape == (T, 63), f"m_τ shape {m_tau.shape}"
    print("=" * 60)
    print(f"  저장 완료: {out_dir}")
    print(f"  eod5{eod5.shape} eod_ohlcv{eod_ohlcv.shape} m_tau{m_tau.shape} "
          f"mask 유효율={mask_data.mean():.3f}")
    print(f"  assert 통과: eod5[...,4]==price, eod_ohlcv[...,3]==price, price>0, "
          f"gt finite, mask no-NaN, m_τ=(T,63)")
    print(f"  다음 단계: 이 폴더({out_dir})를 드라이브에 업로드 → 코랩 셀 3.5(A)에서 복사")
    print("=" * 60)


if __name__ == "__main__":
    main()
