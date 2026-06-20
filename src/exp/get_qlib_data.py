"""qlib cn_data(opensource) 로컬 다운로드 — CSI300 데이터셋 빌드 준비 1단계.

data build.md §0/§6 Step0: build_dataset.py 가 읽을 qlib 중국 일봉 데이터를 받는다.
MASTER opensource 와 동일 소스(qlib 공개 cn_data)를 쓰며, MASTER 배포 pickle 과 섞지 않는다.

사용:
  python get_qlib_data.py --target-dir ~/.qlib/qlib_data/cn_data

전제: pip install pyqlib  (Python 3.8~3.10 권장; numpy<2 환경이 안전).
"""
import os
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", default="~/.qlib/qlib_data/cn_data",
                    help="cn_data 저장 위치 (build_dataset.py --qlib-dir 와 동일하게)")
    ap.add_argument("--interval", default="1d", help="1d(일봉) 권장")
    ap.add_argument("--region", default="cn")
    ap.add_argument("--redownload", action="store_true",
                    help="이미 있어도 다시 받기 (기본은 있으면 skip)")
    args = ap.parse_args()

    target = os.path.expanduser(args.target_dir)
    os.makedirs(os.path.dirname(target.rstrip("/")), exist_ok=True)

    from qlib.tests.data import GetData
    GetData().qlib_data(
        target_dir=target,
        region=args.region,
        interval=args.interval,
        exists_skip=not args.redownload,
    )
    # 간단 검증: 캘린더/인스트루먼트 파일이 생겼는지
    cal = os.path.join(target, "calendars", f"day.txt")
    inst = os.path.join(target, "instruments", "csi300.txt")
    print("=" * 60)
    print(f"  다운로드 위치: {target}")
    print(f"  calendars/day.txt 존재: {os.path.exists(cal)}")
    print(f"  instruments/csi300.txt 존재: {os.path.exists(inst)}")
    if not os.path.exists(inst):
        print("  ⚠️ csi300.txt 가 없습니다. region=cn 데이터가 맞는지 확인하세요.")
    print("  다음 단계: python build_dataset.py --qlib-dir <위 경로> ...")
    print("=" * 60)


if __name__ == "__main__":
    main()
