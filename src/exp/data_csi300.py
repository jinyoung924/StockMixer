"""
CSI300 데이터 로더 — build_dataset.py 가 만든 StockMixer `(N,T,F)` pkl 계약을 읽는다.

dataset/CSI300/ (전 arm 공용, mask/gt/price 는 바이트 단위 동일):
  eod_data_5.pkl      (N, T, 5)    baseline 입력 (종가-MA, close=idx 4)
  eod_data_ohlcv.pkl  (N, T, 5)    add/concat/gating 입력 (OHLCV, close=idx 3)
  price_data.pkl      (N, T)       종목별 정규화 close 레벨 (get_loss 의 base_price)
  gt_data.pkl         (N, T)       forward return 라벨 (raw, StockMixer 로직)
  mask_data.pkl       (N, T)       1=유효, 0=결측/비유니버스
  m_tau.pkl           (T, 63)      시장레짐 (그날 단일 벡터, 전 종목 broadcast)
  meta.pkl            stock_order, date_order, valid_index, test_index, steps

feature(close5 or ohlcv)에 따라 eod 파일만 바꾸고 나머지는 공유 → C1·C2·C3 짝 비교 성립.
"""
import os
import pickle
import numpy as np

# feature 이름 -> (파일, close 채널 인덱스)
FEATURE_FILES = {
    "close5": ("eod_data_5.pkl", 4),
    "ohlcv":  ("eod_data_ohlcv.pkl", 3),
}


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csi300(data_path, feature="ohlcv", use_m_tau=True, verbose=True):
    """CSI300 pkl 적재 → train_exp 가 쓰는 dict 반환.

    feature   : 'close5'(baseline) | 'ohlcv'(add/concat/gating)
    use_m_tau : concat/gating 이면 True (m_tau 반환). baseline/add 는 False(None).
    """
    if feature not in FEATURE_FILES:
        raise ValueError(f"알 수 없는 feature='{feature}'. 사용 가능: {list(FEATURE_FILES)}")
    eod_file, close_idx = FEATURE_FILES[feature]
    base = os.path.join(data_path, "CSI300")

    eod_data = np.asarray(_load(os.path.join(base, eod_file)), dtype=np.float32)
    price_data = np.asarray(_load(os.path.join(base, "price_data.pkl")), dtype=np.float32)
    gt_data = np.asarray(_load(os.path.join(base, "gt_data.pkl")), dtype=np.float32)
    mask_data = np.asarray(_load(os.path.join(base, "mask_data.pkl")), dtype=np.float32)
    meta = _load(os.path.join(base, "meta.pkl"))

    m_tau = None
    if use_m_tau:
        m_tau = np.asarray(_load(os.path.join(base, "m_tau.pkl")), dtype=np.float32)
        assert m_tau.shape[0] == mask_data.shape[1], \
            f"m_tau days({m_tau.shape[0]}) != T({mask_data.shape[1]})"

    # 빌드 계약 검증 (data build.md §8): close 채널 == price_data
    assert np.allclose(eod_data[..., close_idx], price_data, atol=1e-4), \
        f"eod_data[...,{close_idx}] != price_data ({feature} close 채널 규약 위반)"

    out = dict(
        eod_data=eod_data, price_data=price_data, gt_data=gt_data,
        mask_data=mask_data, m_tau=m_tau,
        valid_index=int(meta["valid_index"]), test_index=int(meta["test_index"]),
        steps=int(meta.get("steps", 1)),
        stock_order=meta.get("stock_order"), date_order=meta.get("date_order"),
        m_dim=(m_tau.shape[1] if m_tau is not None else None),
    )
    if verbose:
        print(f"[CSI300] stocks={eod_data.shape[0]} days={eod_data.shape[1]} "
              f"F={eod_data.shape[2]} (feature={feature}, close_idx={close_idx}) "
              f"m_dim={out['m_dim']} valid={out['valid_index']} test={out['test_index']} "
              f"steps={out['steps']}")
    return out
