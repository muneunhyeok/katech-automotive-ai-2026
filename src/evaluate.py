"""
교차 평가 — 여러 split에서 동일 모델을 평가해 누수 프리미엄을 측정

한 개의 숫자만 보면 그것이 부풀려진 값인지 알 수 없습니다.
오염된 split과 교정된 split에서 같은 모델을 평가하면,
그 격차가 곧 "누수가 점수를 얼마나 올렸는가"의 하한이 됩니다.

★ imgsz는 반드시 학습 때와 동일해야 합니다.
  다르면 2.5%p 수준의 허수 차이가 발생합니다.

사용법:
    python evaluate.py clean_final.pt
"""

import sys
from ultralytics import YOLO

IMGSZ = 1536  # 학습과 반드시 동일

# (data.yaml, split, 설명)
TARGETS = [
    ("data/data.yaml",     "val",  "원본 split (누수 의심)"),
    ("data_seq/data.yaml", "val",  "교정 split — 모델 선택용"),
    ("data_seq/data.yaml", "test", "교정 split — 최종 보고용 ★"),
]


def main(weights: str):
    model = YOLO(weights)
    print(f"weights: {weights}  |  imgsz: {IMGSZ}\n")
    print(f"{'split':<38}{'mAP50':>9}{'mAP50-95':>11}   클래스별 AP50-95")
    print("-" * 100)

    for yml, split, note in TARGETS:
        try:
            r = model.val(data=yml, split=split, imgsz=IMGSZ,
                          batch=8, workers=2, device=0,
                          conf=0.001, iou=0.7, max_det=300,
                          plots=False, verbose=False)
            per_class = "  ".join(f"{n}:{r.box.maps[i]:.3f}" for i, n in r.names.items())
            label = f"{yml} / {split}"
            print(f"{label:<38}{r.box.map50:>9.4f}{r.box.map:>11.4f}   {per_class}")
            print(f"{'':<38}{note}")
        except Exception as e:
            print(f"{yml} / {split:<10} 실패: {type(e).__name__}: {e}")

    print("\n해석:")
    print("  - 원본 val과 교정 val의 격차가 크면 누수가 점수를 올리고 있었다는 뜻입니다.")
    print("  - 격차가 거의 없다면, 누수는 있었지만 모델이 암기로 점수를 얻지는 않은 것입니다.")
    print("  - 발표/보고에 쓸 숫자는 교정 split의 test 값입니다.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
