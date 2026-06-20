"""
시장레짐 정보 × 주입 구조 비교 실험 — CSI300 / StockMixer 하네스.

검증 목표(두 축을 분리):
  Q1 정보의 가치  : 더 풍부한 정보(alpha158 13피처 + m_τ)가 예측력을 올리는가?
  Q2 구조의 역할  : 같은 정보라도 주입 구조(add/concat/gating)가 결과를 바꾸는가?

4개 arm (입력 피처 × 결합부만 다르고 mask/gt/price·백본·분할은 전부 고정):
  baseline   : 종가 5피처     + AddHead        (CSI300 위의 원본 StockMixer, 절대 출발점)
  add        : alpha158 13피처 + AddHead        (구조 비교 기준, m_τ 미반영)
  concat     : alpha158 13피처 + Concat(m_τ)    (덧셈적 주입; _lin / _mlp 변형)
  gating     : alpha158 13피처 + Gate(m_τ)      (곱셈적 주입, 주 모델)

비교 축:
  C1(피처)  baseline ↔ add          : 결합부 동일, 입력 5→13
  C2(정보)  add ↔ {concat, gating}  : 피처 13 동일, m_τ 주입 여부
  C3(구조)  concat ↔ gating         : 피처·정보·백본 동일, 결합부만
  D1(진단)  add ↔ concat_lin        : 선형 주입은 순위 불변 예상

원본 train.py 의 학습/예측/평가 골격을 계승하되:
  - 라벨은 StockMixer raw forward return, 손실은 원본 get_loss(base_price=price_data).
  - 평가는 evaluator_exp(IC/ICIR/RankIC/RankICIR/prec@10/SR).
  - 결과는 arm×seed 로 experiments/ 에 쌓고 comparison.csv 로 재구성.

사용 예:
  python train_exp.py --arm baseline   --seed 0 --epochs 100
  python train_exp.py --arm add        --seed 0 --epochs 100
  python train_exp.py --arm concat_mlp --seed 0 --epochs 100
  python train_exp.py --arm gating     --seed 0 --beta 2 --epochs 100
"""
import os
import sys
import csv
import glob
import json
import time
import random
import argparse
import datetime
import numpy as np
import torch

# src/exp/ 에서 부모 src 의 원본 model.py(get_loss)를 import 하기 위한 경로 추가
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from model import get_loss                 # 원본 손실(MSE + α·pairwise rank), 그대로 계승
from model_exp import StockMixerExp
from evaluator_exp import evaluate
from data_csi300 import load_csi300
from final_layers import list_modes, M_DIM


# arm -> (입력 피처 수, 결합부 combine_mode, m_τ 사용 여부)
# baseline/add 는 같은 AddHead 를 쓰되 입력 피처만 5 vs 13 (C1).
# concat/gating 은 13피처 고정, m_τ 주입 구조만 다름 (C3).
ARM_CONFIG = {
    "baseline":   dict(feature_set=5,  combine_mode="add",        use_m_tau=False),
    "add":        dict(feature_set=13, combine_mode="add",        use_m_tau=False),
    "concat_lin": dict(feature_set=13, combine_mode="concat_lin", use_m_tau=True),
    "concat_mlp": dict(feature_set=13, combine_mode="concat_mlp", use_m_tau=True),
    "gating":     dict(feature_set=13, combine_mode="gate",       use_m_tau=True),
}


def set_seed(seed=0):
    # 시드 스윕(0~4) 짝 비교를 위해 모든 난수원 통일 + 결정론 설정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="baseline",
                    help="baseline | add | concat_lin | concat_mlp | gating")
    ap.add_argument("--seed", type=int, default=0, help="재현용 시드 (0~4 스윕 권장)")
    ap.add_argument("--beta", type=float, default=2.0,
                    help="gating 게이팅 온도 (작을수록 강함; β 스윕 {1,2,5,10})")
    ap.add_argument("--hidden", type=int, default=None, help="concat_mlp 은닉 차원 (기본 feat_dim)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--alpha", type=float, default=0.1, help="pairwise rank loss 가중")
    ap.add_argument("--lookback", type=int, default=16)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--market-num", type=int, default=20,
                    help="NoGraphMixer hidden(cross-stock mixing 차원). m_τ 와 무관, 전 arm 고정 20")
    ap.add_argument("--steps", type=int, default=None,
                    help="forward return 호라이즌(미지정 시 meta.steps 사용)")
    ap.add_argument("--data-path", default="../../dataset",
                    help="dataset 폴더 (src/exp 기준 기본값). CSI300/ 하위를 읽는다")
    ap.add_argument("--exp-root", default="../../experiments")
    ap.add_argument("--tag", default="", help="실험 메모(선택)")
    ap.add_argument("--list", action="store_true", help="등록된 arm / combine_mode 출력")
    args = ap.parse_args()

    if args.list:
        print("등록된 arm:", ", ".join(ARM_CONFIG))
        print("등록된 combine_mode:", ", ".join(list_modes()))
        return

    if args.arm not in ARM_CONFIG:
        raise ValueError(f"알 수 없는 arm='{args.arm}'. 사용 가능: {list(ARM_CONFIG)}")
    acfg = ARM_CONFIG[args.arm]

    set_seed(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ---- 데이터 ----
    d = load_csi300(args.data_path, feature_set=acfg["feature_set"],
                    use_m_tau=acfg["use_m_tau"])
    eod_data, mask_data = d["eod_data"], d["mask_data"]
    gt_data, price_data, m_tau = d["gt_data"], d["price_data"], d["m_tau"]
    stock_num = eod_data.shape[0]
    trade_dates = eod_data.shape[1]
    fea_num = eod_data.shape[-1]
    valid_index, test_index = d["valid_index"], d["test_index"]
    steps = args.steps if args.steps is not None else d["steps"]
    lookback = args.lookback
    market_dim = d["m_dim"] or M_DIM   # head 의 m_τ 입력 차원(63)

    print(f"arm={args.arm} | stocks={stock_num} days={trade_dates} F={fea_num} "
          f"m_τ={'on' if acfg['use_m_tau'] else 'off'} "
          f"(valid={valid_index}, test={test_index}, steps={steps})")

    # ---- 모델 ----
    model = StockMixerExp(
        stocks=stock_num, time_steps=lookback, channels=fea_num,
        market=args.market_num, scale=args.scale,
        combine_mode=acfg["combine_mode"], beta=args.beta,
        hidden=args.hidden, market_dim=market_dim,
    ).to(device)
    info = model.describe()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- 실험 폴더 ----
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{args.arm}_CSI300_s{args.seed}_{stamp}"
    exp_dir = os.path.join(args.exp_root, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    out = lambda name: os.path.join(exp_dir, f"{exp_id}_{name}")

    config = dict(
        exp_id=exp_id, arm=args.arm, combine_mode=acfg["combine_mode"],
        feature_set=acfg["feature_set"], use_m_tau=acfg["use_m_tau"],
        market="CSI300", seed=args.seed,
        epochs=args.epochs, lr=args.lr, alpha=args.alpha,
        beta=args.beta, hidden=args.hidden,
        lookback=lookback, channels=fea_num, market_num=args.market_num,
        m_dim=market_dim, scale=args.scale, steps=steps,
        stock_num=stock_num, device=str(device),
        total_params=info["total_params"], head_params=info["head_params"],
        tag=args.tag, timestamp=stamp,
    )
    with open(out("config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"  실험 시작: {exp_id}")
    print(f"  arm={args.arm} combine={acfg['combine_mode']} seed={args.seed} device={device}")
    print(f"  파라미터 총 {info['total_params']:,} (head {info['head_params']:,})")
    print("=" * 60)

    # ---- 배치 유틸 (원본 train.py 계승) ----
    batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

    def get_batch(offset):
        sl = lookback
        mb = mask_data[:, offset: offset + sl + steps]
        mb = np.min(mb, axis=1)                                  # 윈도우+steps 전체 유효한 종목만
        # m_τ: 결정(예측)일 = 입력 윈도우 마지막 날(offset+sl-1)의 시장레짐 벡터.
        # 라벨일(offset+sl+steps-1)이 아니라 마지막 관측일을 써 lookahead 누수를 피한다.
        mt = m_tau[offset + sl - 1] if m_tau is not None else None
        return (
            eod_data[:, offset:offset + sl, :],
            np.expand_dims(mb, axis=1),
            np.expand_dims(price_data[:, offset + sl - 1], axis=1),     # base_price
            np.expand_dims(gt_data[:, offset + sl + steps - 1], axis=1),
            mt,
        )

    def to_dev(db, mb, pb, gb, mt):
        t = lambda x: torch.Tensor(x).to(device)
        return t(db), t(mb), t(pb), t(gb), (t(mt) if mt is not None else None)

    def validate(start_index, end_index):
        with torch.no_grad():
            n = end_index - start_index
            pred = np.zeros([stock_num, n], dtype=float)
            gt = np.zeros([stock_num, n], dtype=float)
            msk = np.zeros([stock_num, n], dtype=float)
            loss = reg_loss = rank_loss = 0.0
            base = start_index - lookback - steps + 1
            for off in range(base, end_index - lookback - steps + 1):
                db, mb, pb, gb, mt = to_dev(*get_batch(off))
                prediction = model(db, mt)
                cl, crl, crk, cur_rr = get_loss(prediction, gb, pb, mb, stock_num, args.alpha)
                loss += cl.item(); reg_loss += crl.item(); rank_loss += crk.item()
                idx = off - base
                pred[:, idx] = cur_rr[:, 0].cpu()
                gt[:, idx] = gb[:, 0].cpu()
                msk[:, idx] = mb[:, 0].cpu()
            loss /= n
            perf = evaluate(pred, gt, msk)
        return loss, perf

    # ---- 히스토리 csv ----
    hist_path = out("history.csv")
    metric_keys = ["IC", "ICIR", "RankIC", "RankICIR", "prec_10", "sharpe5"]
    hist_fields = (["epoch", "train_loss", "val_loss", "test_loss"]
                   + [f"val_{k}" for k in metric_keys]
                   + [f"test_{k}" for k in metric_keys])
    hist_file = open(hist_path, "w", newline="")
    hist_writer = csv.DictWriter(hist_file, fieldnames=hist_fields)
    hist_writer.writeheader()

    best_valid_loss = np.inf
    best = None
    t_start = time.time()

    for epoch in range(args.epochs):
        np.random.shuffle(batch_offsets)
        tra_loss = 0.0
        n_batch = valid_index - lookback - steps + 1
        model.train()
        for j in range(n_batch):
            db, mb, pb, gb, mt = to_dev(*get_batch(batch_offsets[j]))
            optimizer.zero_grad()
            prediction = model(db, mt)
            cl, _, _, _ = get_loss(prediction, gb, pb, mb, stock_num, args.alpha)
            cl.backward()
            optimizer.step()
            tra_loss += cl.item()
        tra_loss /= n_batch

        model.eval()
        val_loss, val_perf = validate(valid_index, test_index)
        test_loss, test_perf = validate(test_index, trade_dates)

        row = dict(epoch=epoch + 1, train_loss=tra_loss, val_loss=val_loss, test_loss=test_loss)
        for k in metric_keys:
            row[f"val_{k}"] = val_perf[k]
            row[f"test_{k}"] = test_perf[k]
        hist_writer.writerow(row)
        hist_file.flush()

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best = dict(epoch=epoch + 1, val_loss=val_loss,
                        val_perf=val_perf, test_perf=test_perf)
            torch.save({"model_state": model.state_dict(), "config": config,
                        "epoch": epoch + 1}, out("best_model.pt"))

        print(f"[{epoch+1:3d}/{args.epochs}] train={tra_loss:.2e} val={val_loss:.2e} "
              f"test={test_loss:.2e} | test IC={test_perf['IC']:.4f} "
              f"RankIC={test_perf['RankIC']:.4f} prec@10={test_perf['prec_10']:.4f} "
              f"SR={test_perf['sharpe5']:.4f}")

    hist_file.close()
    elapsed = time.time() - t_start

    # ---- best 요약 + 비교표 재구성 ----
    bt, bv = best["test_perf"], best["val_perf"]
    summary = dict(
        exp_id=exp_id, arm=args.arm, combine_mode=acfg["combine_mode"],
        feature_set=acfg["feature_set"], use_m_tau=acfg["use_m_tau"],
        seed=args.seed, beta=args.beta, hidden=args.hidden,
        epochs=args.epochs, best_epoch=best["epoch"],
        best_val_loss=round(best["val_loss"], 6),
        val_IC=round(bv["IC"], 4), val_RankIC=round(bv["RankIC"], 4),
        test_IC=round(bt["IC"], 4), test_ICIR=round(bt["ICIR"], 4),
        test_RankIC=round(bt["RankIC"], 4), test_RankICIR=round(bt["RankICIR"], 4),
        test_prec10=round(bt["prec_10"], 4), test_SR=round(bt["sharpe5"], 4),
        total_params=info["total_params"], head_params=info["head_params"],
        minutes=round(elapsed / 60, 1), tag=args.tag, timestamp=stamp,
    )
    with open(out("best_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # comparison.csv : 모든 best_summary.json 을 모아 exp_id 정렬로 매번 재작성(append 아님)
    os.makedirs(args.exp_root, exist_ok=True)
    cmp_path = os.path.join(args.exp_root, "comparison.csv")
    all_rows = []
    for p in glob.glob(os.path.join(args.exp_root, "*", "*best_summary.json")):
        try:
            with open(p) as jf:
                all_rows.append(json.load(jf))
        except (OSError, json.JSONDecodeError):
            pass
    all_rows.sort(key=lambda r: r.get("exp_id", ""))
    fieldnames = list(summary.keys())
    for r in all_rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(cmp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(all_rows)

    print("=" * 60)
    print(f"  완료: best epoch {best['epoch']} (val_loss={best['val_loss']:.2e})")
    print(f"  test IC={bt['IC']:.4f} ICIR={bt['ICIR']:.4f} RankIC={bt['RankIC']:.4f} "
          f"prec@10={bt['prec_10']:.4f} SR={bt['sharpe5']:.4f}")
    print(f"  소요 {summary['minutes']} 분 | 저장 {exp_dir}")
    print(f"  비교표 재구성: {cmp_path} (총 {len(all_rows)}개 실험)")
    print("=" * 60)


if __name__ == "__main__":
    main()
