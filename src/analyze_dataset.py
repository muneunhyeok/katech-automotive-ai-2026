"""
데이터셋 진단 — 클래스별 객체 크기 분포 측정

하이퍼파라미터를 건드리기 전에 "무엇이 병목인지"를 먼저 측정합니다.
이 스크립트 하나가 imgsz를 최우선 레버로 선택한 근거가 되었습니다.

실제 측정 결과 (docs/03-object-detection.md):
    pedestrian 크기 중앙값 ~31px, 32px 미만 비율 55%  ← 유일한 실질 병목
    vehicle / cycle 은 32px 미만 0%

사용법:
    python analyze_dataset.py data/images/train data/labels/train
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

NAMES = {0: "vehicle", 1: "pedestrian", 2: "cycle"}
SMALL = 32  # COCO 소형 객체 기준 (픽셀)


def main(img_dir: str, label_dir: str):
    img_dir, label_dir = Path(img_dir), Path(label_dir)
    sizes = defaultdict(list)
    counts = defaultdict(int)
    n_images = 0
    missing = []

    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        n_images += 1
        lb = (label_dir / img.stem).with_suffix(".txt")
        if not lb.exists():
            missing.append(img.name)
            continue

        W, H = Image.open(img).size
        for line in lb.read_text().splitlines():
            if not line.strip():
                continue
            c, _x, _y, w, h = line.split()[:5]
            c = int(c)
            # YOLO 포맷은 정규화 좌표 → 픽셀로 환산
            side = np.sqrt(float(w) * W * float(h) * H)  # 면적의 제곱근을 대표 크기로
            sizes[c].append(side)
            counts[c] += 1

    print(f"이미지 {n_images}장 / 라벨 {n_images - len(missing)}개")
    if missing:
        print(f"[경고] 라벨 없는 이미지 {len(missing)}장: {missing[:10]}")

    print(f"\n{'클래스':<12}{'인스턴스':>8}{'중앙값(px)':>12}{'평균(px)':>10}{'<32px 비율':>12}")
    print("-" * 56)
    for c in sorted(sizes):
        a = np.array(sizes[c])
        print(f"{NAMES.get(c, c):<12}{len(a):>8}{np.median(a):>12.1f}"
              f"{a.mean():>10.1f}{(a < SMALL).mean() * 100:>11.1f}%")

    print("\n해석:")
    print("  - <32px 비율이 높은 클래스가 있으면 입력 해상도(imgsz) 상향이 최우선 레버입니다.")
    print("  - 인스턴스 수가 적은 클래스는 지표 노이즈가 큽니다. 소수점 셋째 자리에 의미를 두지 마세요.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
