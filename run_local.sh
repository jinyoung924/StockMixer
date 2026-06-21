#!/usr/bin/env bash
# 로컬 맥 스모크 테스트 — 코랩에 올리기 전에 "학습이 제대로 도는지"만 빠르게 확인한다.
# cv_02_실험환경.py [셀 4]와 같은 train_exp.py CLI 를 그대로 호출하되,
#   · epochs 를 작게(기본 2)            → 파이프라인만 확인
#   · seed 1개, arm 일부만               → 시간 절약
#   · GPU 없으면 자동 CPU(맥)            → train_exp.py 가 cuda-or-cpu 로 폴백
#   · --data-path 기본(../../dataset)    → dataset/CSI300/*.pkl 을 그대로 읽음
#
# 사용:
#   ./run_local.sh                 # 기본: arm=baseline,gating / epochs=2 / seed=0
#   EPOCHS=5 ./run_local.sh        # epochs 바꾸기
#   ARMS="add concat_mlp" ./run_local.sh
#   SEED=1 STEPS=5 ./run_local.sh
set -euo pipefail

# 이 스크립트 위치 기준으로 경로 고정 (어디서 실행해도 동작)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${REPO_DIR}/src/exp"

# ── 조정 가능한 변수 (환경변수로 덮어쓰기) ──
EPOCHS="${EPOCHS:-2}"                       # 스모크 테스트는 2~5면 충분
SEED="${SEED:-0}"
STEPS="${STEPS:-5}"                         # forward return 호라이즌
BETA="${BETA:-2.0}"                         # gating 온도 (다른 arm 엔 무시됨)
ARMS="${ARMS:-baseline gating}"            # 전체: baseline add concat_lin concat_mlp gating
EXP_ROOT="${EXP_ROOT:-${REPO_DIR}/experiments_local}"   # 결과 저장(드라이브 X, 로컬)
PY="${PY:-python3}"

echo "=================================================="
echo " 로컬 스모크 테스트"
echo "   ARMS=${ARMS}"
echo "   EPOCHS=${EPOCHS}  SEED=${SEED}  STEPS=${STEPS}  BETA=${BETA}"
echo "   데이터: ${REPO_DIR}/dataset/CSI300"
echo "   결과:   ${EXP_ROOT}"
echo "=================================================="

# 데이터 존재 확인 (없으면 바로 알려주고 종료)
if [ ! -f "${REPO_DIR}/dataset/CSI300/eod_data_ohlcv.pkl" ]; then
  echo "ERROR: dataset/CSI300/*.pkl 을 찾지 못했습니다: ${REPO_DIR}/dataset/CSI300" >&2
  exit 1
fi

for arm in ${ARMS}; do
  echo ""
  echo ">>> arm=${arm} | seed=${SEED} | steps=${STEPS} | epochs=${EPOCHS} 시작"
  ( cd "${EXP_DIR}" && "${PY}" train_exp.py \
      --arm "${arm}" --seed "${SEED}" --steps "${STEPS}" \
      --beta "${BETA}" --epochs "${EPOCHS}" \
      --exp-root "${EXP_ROOT}" )
done

echo ""
echo "완료. 결과 요약/비교표:"
echo "  ${EXP_ROOT}/comparison.csv"
echo "  ${EXP_ROOT}/<exp_id>/<exp_id>_history.csv  (에폭별 곡선)"
