"""
확장 평가기 — 원본 evaluator.py 의 RIC 함정을 교정하고 순위 지표를 추가한다.

⚠️ 원본 함정: StockMixer evaluator 의 `RIC = mean(ic)/std(ic)` 에서 ic 는 Pearson 이므로
   이는 사실 ICIR(IC 정보비율)이지 순위 IC(Spearman)가 아니다.
   본 실험 문서의 RIC(RankIC)는 Spearman 으로 정의되므로 별도 구현이 필요하다.

보고 지표:
  IC        일별 Pearson(pred, gt) 평균            — 예측-현실 선형 일치도
  ICIR      mean(IC)/std(IC)  (= 원본의 "RIC")     — IC 안정성(정보비율)
  RankIC    일별 Spearman(pred, gt) 평균  [추가]    — 순서 일치도(문서의 "RIC"), 이상치 강건
  RankICIR  mean(RankIC)/std(RankIC)      [추가]    — 순위 IC 안정성
  prec_10   예측 상위10 중 실제 상승 비율            — 상위 픽 적중률 (원본 그대로)
  sharpe5   상위5 포트 일수익 평균/표준편차 ×15.87    — 위험 대비 안정성(SR) (원본 그대로)
  mse       참고용

IC/RankIC 는 매일 '마스크된 유효 종목'만으로 계산한다(원본의 pred*mask 후 전체상관 편향 제거).
prec_10/sharpe5 는 원본 로직을 바이트 단위로 보존해 기존 StockMixer 수치와 비교 가능하게 둔다.
"""
import numpy as np
import pandas as pd


def _daily_corr(pred_col, gt_col, valid, method):
    """그날 유효 종목만으로 Pearson/Spearman 상관. 유효<2 또는 분산0이면 NaN."""
    if valid.sum() < 2:
        return np.nan
    p = pd.Series(pred_col[valid])
    g = pd.Series(gt_col[valid])
    if p.std() == 0 or g.std() == 0:
        return np.nan
    return p.corr(g, method=method)


def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, "shape mis-match"
    performance = {}
    performance["mse"] = np.linalg.norm((prediction - ground_truth) * mask) ** 2 / np.sum(mask)

    ic, rank_ic = [], []
    sharpe_li5 = []
    prec_10 = []

    n_days = prediction.shape[1]
    n_stocks = prediction.shape[0]

    for i in range(n_days):
        valid = mask[:, i] >= 0.5
        # ---- IC(Pearson) / RankIC(Spearman) : 유효 종목만 ----
        ic.append(_daily_corr(prediction[:, i], ground_truth[:, i], valid, "pearson"))
        rank_ic.append(_daily_corr(prediction[:, i], ground_truth[:, i], valid, "spearman"))

        # ---- prec@10 / sharpe5 : 원본 evaluator 로직 보존 ----
        rank_pre = np.argsort(prediction[:, i])
        pre_top5, pre_top10 = set(), set()
        for j in range(1, n_stocks + 1):
            cur_rank = rank_pre[-j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)

        real_ret_top5 = sum(ground_truth[p][i] for p in pre_top5) / 5
        sharpe_li5.append(real_ret_top5)

        prec = sum((ground_truth[p][i] >= 0) for p in pre_top10)
        prec_10.append(prec / 10)

    ic = np.array(ic, dtype=float)
    rank_ic = np.array(rank_ic, dtype=float)
    sharpe_li5 = np.array(sharpe_li5, dtype=float)

    performance["IC"] = np.nanmean(ic)
    performance["ICIR"] = np.nanmean(ic) / np.nanstd(ic)
    performance["RankIC"] = np.nanmean(rank_ic)
    performance["RankICIR"] = np.nanmean(rank_ic) / np.nanstd(rank_ic)
    performance["prec_10"] = np.mean(prec_10)
    performance["sharpe5"] = (np.mean(sharpe_li5) / np.std(sharpe_li5)) * 15.87
    return performance
