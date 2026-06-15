"""
결합부(combine) 메커니즘 비교 실험용 학습 스크립트.

원본 train.py 의 학습/평가 로직은 그대로 따르되, 다음을 추가한다:
  1) --combine-mode 로 결합부를 선택 (add / gate / film / attn)
  2) 검증 손실 기준 best 모델 가중치를 .pt 로 저장
  3) 에폭별 지표를 history.csv 로 저장 (train vs valid/test 격차 = 과적합 진단용)
  4) 실험 설정을 config.json 으로 저장
  5) 모든 실험을 comparison.csv 한 파일에 누적해 비교 가능

(데이터셋 2종) × (결합 4종) = 8개 설정을 --market / --combine-mode 토글로 전환한다.

사용 예 (코랩 셀에서):
  !cd "{REPO_DIR}/src/exp" && python train_exp.py --combine-mode add  --market NASDAQ --epochs 100
  !cd "{REPO_DIR}/src/exp" && python train_exp.py --combine-mode gate --market SP500  --beta 5 --epochs 100
  !cd "{REPO_DIR}/src/exp" && python train_exp.py --combine-mode attn --market NASDAQ --d-attn 16 --epochs 100

결과는 기본적으로 ../../experiments/ 아래에 쌓인다 (--exp-root 로 변경 가능).
"""
import os
import sys
import csv
import json
import time
import random
import argparse
import datetime
import numpy as np
import torch
import pickle

# 이 파일은 src/exp/ 안에 있으므로, 부모 src 폴더를 import 경로에 추가해
# 원본 evaluator.py / model.py / load_data.py 를 그대로 쓸 수 있게 한다.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from evaluator import evaluate
from model import get_loss
from model_exp import StockMixerExp
from final_layers import list_modes


# 데이터셋별 분할 인덱스 + stock mixing 의 market_num (paper Table 1 기준)
# market_num 은 NoGraphMixer 의 hidden dim (결합부 m_tau 차원과 무관, 혼동 주의)
MARKET_CONFIG = {
    "NASDAQ": dict(valid_index=756, test_index=1008, market_num=20),
    "SP500":  dict(valid_index=1006, test_index=1259, market_num=8),
}


def set_seed(seed=12345678):
    random.seed(seed)
    np.random.seed(123456789)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_path, market_name, steps):
    """원본 train.py 와 동일한 방식으로 데이터를 적재."""
    if market_name == "SP500":
        data = np.load(os.path.join(data_path, "SP500", "SP500.npy"))
        data = data[:, 915:, :]
        price_data = data[:, :, -1]
        mask_data = np.ones((data.shape[0], data.shape[1]))
        eod_data = data
        gt_data = np.zeros((data.shape[0], data.shape[1]))
        for t in range(data.shape[0]):
            for r in range(1, data.shape[1]):
                gt_data[t][r] = (data[t][r][-1] - data[t][r - steps][-1]) / \
                                data[t][r - steps][-1]
    else:
        base = os.path.join(data_path, market_name)
        with open(os.path.join(base, "eod_data.pkl"), "rb") as f:
            eod_data = pickle.load(f)
        with open(os.path.join(base, "mask_data.pkl"), "rb") as f:
            mask_data = pickle.load(f)
        with open(os.path.join(base, "gt_data.pkl"), "rb") as f:
            gt_data = pickle.load(f)
        with open(os.path.join(base, "price_data.pkl"), "rb") as f:
            price_data = pickle.load(f)
    return eod_data, mask_data, gt_data, price_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combine-mode", default="add",
                    help="add | gate | film | attn")
    ap.add_argument("--market", default="NASDAQ", help="NASDAQ | SP500")
    ap.add_argument("--beta", type=float, default=5.0,
                    help="gate/film 게이팅 온도 (작을수록 강함)")
    ap.add_argument("--d-attn", type=int, default=16, help="attn 모드 어텐션 차원")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lookback", type=int, default=16)
    ap.add_argument("--market-dim", type=int, default=None,
                    help="stock mixing 의 hidden m (미지정 시 데이터셋 기본값)")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--data-path", default="../../dataset",
                    help="dataset 폴더 (src/exp 기준 기본값)")
    ap.add_argument("--exp-root", default="../../experiments",
                    help="결과 저장 루트 (드라이브 경로로 바꿔도 됨)")
    ap.add_argument("--tag", default="", help="실험 메모(선택)")
    ap.add_argument("--list", action="store_true", help="등록된 combine_mode 목록만 출력")
    args = ap.parse_args()

    if args.list:
        print("등록된 combine_mode:", ", ".join(list_modes()))
        return

    set_seed()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ---- 데이터 ----
    fea_num = 5
    eod_data, mask_data, gt_data, price_data = load_dataset(
        args.data_path, args.market, args.steps)
    stock_num = eod_data.shape[0]          # 데이터에서 자동 추출 (원본의 하드코딩 footgun 방지)
    trade_dates = mask_data.shape[1]
    mc = MARKET_CONFIG.get(args.market, MARKET_CONFIG["NASDAQ"])
    valid_index, test_index = mc["valid_index"], mc["test_index"]
    market_dim = args.market_dim if args.market_dim is not None else mc["market_num"]
    lookback = args.lookback
    steps = args.steps
    print(f"data: stocks={stock_num}, days={trade_dates}, F={eod_data.shape[-1]} "
          f"(market_num={market_dim}, valid={valid_index}, test={test_index})")

    # ---- 모델 ----
    model = StockMixerExp(
        stocks=stock_num, time_steps=lookback, channels=fea_num,
        market=market_dim, scale=args.scale,
        combine_mode=args.combine_mode, beta=args.beta, d_attn=args.d_attn,
    ).to(device)
    info = model.describe()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- 실험 폴더 ----
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{args.combine_mode}_{args.market}_{stamp}"
    exp_dir = os.path.join(args.exp_root, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    config = dict(
        exp_id=exp_id, combine_mode=args.combine_mode,
        market=args.market, epochs=args.epochs, lr=args.lr, alpha=args.alpha,
        beta=args.beta, d_attn=args.d_attn,
        lookback=lookback, market_dim=market_dim, scale=args.scale,
        steps=steps, stock_num=stock_num, device=str(device),
        total_params=info["total_params"], head_params=info["head_params"],
        tag=args.tag, timestamp=stamp,
    )
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"  실험 시작: {exp_id}")
    print(f"  combine_mode={args.combine_mode} | market={args.market} "
          f"| device={device}")
    print(f"  파라미터 총 {info['total_params']:,} (head {info['head_params']:,})")
    print("=" * 60)

    # ---- 배치 유틸 (원본 train.py 와 동일) ----
    batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

    def get_batch(offset):
        sl = lookback
        mb = mask_data[:, offset: offset + sl + steps]
        mb = np.min(mb, axis=1)
        return (
            eod_data[:, offset:offset + sl, :],
            np.expand_dims(mb, axis=1),
            np.expand_dims(price_data[:, offset + sl - 1], axis=1),
            np.expand_dims(gt_data[:, offset + sl + steps - 1], axis=1),
        )

    def validate(start_index, end_index):
        with torch.no_grad():
            n = end_index - start_index
            pred = np.zeros([stock_num, n], dtype=float)
            gt = np.zeros([stock_num, n], dtype=float)
            msk = np.zeros([stock_num, n], dtype=float)
            loss = reg_loss = rank_loss = 0.0
            base = start_index - lookback - steps + 1
            for off in range(base, end_index - lookback - steps + 1):
                db, mb, pb, gb = map(lambda x: torch.Tensor(x).to(device), get_batch(off))
                prediction = model(db, mb)   # mask 전달 → masked m_tau
                cl, crl, crk, cur_rr = get_loss(prediction, gb, pb, mb, stock_num, args.alpha)
                loss += cl.item(); reg_loss += crl.item(); rank_loss += crk.item()
                idx = off - base
                pred[:, idx] = cur_rr[:, 0].cpu()
                gt[:, idx] = gb[:, 0].cpu()
                msk[:, idx] = mb[:, 0].cpu()
            loss /= n
            perf = evaluate(pred, gt, msk)
        return loss, perf

    # ---- 히스토리 csv 준비 ----
    hist_path = os.path.join(exp_dir, "history.csv")
    hist_fields = ["epoch", "train_loss", "val_loss", "test_loss",
                   "val_IC", "val_RIC", "val_prec10", "val_SR",
                   "test_IC", "test_RIC", "test_prec10", "test_SR"]
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
            db, mb, pb, gb = map(lambda x: torch.Tensor(x).to(device),
                                 get_batch(batch_offsets[j]))
            optimizer.zero_grad()
            prediction = model(db, mb)   # mask 전달 → masked m_tau
            cl, _, _, _ = get_loss(prediction, gb, pb, mb, stock_num, args.alpha)
            cl.backward()
            optimizer.step()
            tra_loss += cl.item()
        tra_loss /= n_batch

        model.eval()
        val_loss, val_perf = validate(valid_index, test_index)
        test_loss, test_perf = validate(test_index, trade_dates)

        hist_writer.writerow(dict(
            epoch=epoch + 1, train_loss=tra_loss, val_loss=val_loss, test_loss=test_loss,
            val_IC=val_perf["IC"], val_RIC=val_perf["RIC"],
            val_prec10=val_perf["prec_10"], val_SR=val_perf["sharpe5"],
            test_IC=test_perf["IC"], test_RIC=test_perf["RIC"],
            test_prec10=test_perf["prec_10"], test_SR=test_perf["sharpe5"],
        ))
        hist_file.flush()

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best = dict(epoch=epoch + 1, val_loss=val_loss,
                        val_perf=val_perf, test_perf=test_perf)
            # best 가중치 저장
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "epoch": epoch + 1,
            }, os.path.join(exp_dir, "best_model.pt"))

        print(f"[{epoch+1:3d}/{args.epochs}] "
              f"train={tra_loss:.2e} val={val_loss:.2e} test={test_loss:.2e} | "
              f"test IC={test_perf['IC']:.4f} RIC={test_perf['RIC']:.4f} "
              f"prec@10={test_perf['prec_10']:.4f} SR={test_perf['sharpe5']:.4f}")

    hist_file.close()
    elapsed = time.time() - t_start

    # ---- best 결과 저장 + 비교표 누적 ----
    bt = best["test_perf"]; bv = best["val_perf"]
    summary = dict(
        exp_id=exp_id, combine_mode=args.combine_mode,
        market=args.market, beta=args.beta, d_attn=args.d_attn,
        epochs=args.epochs, best_epoch=best["epoch"],
        best_val_loss=round(best["val_loss"], 6),
        val_IC=round(bv["IC"], 4), val_RIC=round(bv["RIC"], 4),
        test_IC=round(bt["IC"], 4), test_RIC=round(bt["RIC"], 4),
        test_prec10=round(bt["prec_10"], 4), test_SR=round(bt["sharpe5"], 4),
        total_params=info["total_params"], head_params=info["head_params"],
        minutes=round(elapsed / 60, 1), tag=args.tag, timestamp=stamp,
    )
    with open(os.path.join(exp_dir, "best_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # comparison.csv 누적 (없으면 헤더 생성, 있으면 한 줄 추가)
    os.makedirs(args.exp_root, exist_ok=True)
    cmp_path = os.path.join(args.exp_root, "comparison.csv")
    write_header = not os.path.exists(cmp_path)
    with open(cmp_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary)

    print("=" * 60)
    print(f"  완료: best epoch {best['epoch']} (val_loss={best['val_loss']:.2e})")
    print(f"  test IC={bt['IC']:.4f} RIC={bt['RIC']:.4f} "
          f"prec@10={bt['prec_10']:.4f} SR={bt['sharpe5']:.4f}")
    print(f"  소요 {summary['minutes']} 분")
    print(f"  저장 위치: {exp_dir}")
    print(f"  비교표 누적: {cmp_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()