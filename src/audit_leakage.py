"""
데이터 누수 감사 — perceptual hash 기반 프레임 중복 정량화

같은 주행 영상에서 추출한 데이터셋을 무작위 분할하면, 인접 프레임이
train/val에 나뉘어 들어가 사실상 "본 장면을 다시 보는" 상태가 됩니다.

파일명이나 순번이 아니라 이미지 내용으로 유사도를 재기 위해
16x16 average hash + Hamming distance를 사용합니다.

기준선 해석 (docs/05-data-integrity-audit.md 참고):
  - 거리 <= 10  : near-duplicate. 이 비율이 0%가 되어야 함
  - 인접 번호 쌍 평균 거리와 무작위 쌍 평균 거리를 함께 측정해야
    "이 데이터셋에서 얼마가 정상인지"를 판단할 수 있습니다.

사용법:
    python audit_leakage.py data/images/train data/images/valid
"""

import sys
import random
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = ("*.png", "*.jpg", "*.jpeg")


def list_images(d: Path):
    files = []
    for e in EXTS:
        files.extend(d.glob(e))
    return sorted(files, key=lambda p: (len(p.stem), p.stem))


def ahash(path: Path, size: int = 16) -> np.ndarray:
    """16x16 average hash → 256bit 이진 지문"""
    a = np.asarray(Image.open(path).convert("L").resize((size, size)), np.float32)
    return (a > a.mean()).ravel()


def hamming_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A의 각 행에 대해 B의 모든 행과의 Hamming 거리"""
    return (A[:, None, :] != B[None, :, :]).sum(2)


def baseline_distances(files, hashes, n_random=300):
    """이 데이터셋에서 '가까움'과 '멂'의 기준선을 측정"""
    # 인접 번호 쌍
    adjacent = [np.sum(hashes[i] != hashes[i + 1]) for i in range(len(files) - 1)]
    # 무작위 쌍
    rnd = []
    for _ in range(n_random):
        i, j = random.sample(range(len(files)), 2)
        rnd.append(np.sum(hashes[i] != hashes[j]))
    return float(np.mean(adjacent)), float(np.mean(rnd))


def main(train_dir: str, val_dir: str):
    tr_files = list_images(Path(train_dir))
    va_files = list_images(Path(val_dir))
    if not tr_files or not va_files:
        sys.exit("이미지를 찾지 못했습니다. 경로를 확인하세요.")

    print(f"train {len(tr_files)}장 / val {len(va_files)}장  해시 계산 중...")
    tr_h = np.array([ahash(p) for p in tr_files])
    va_h = np.array([ahash(p) for p in va_files])

    dist = hamming_matrix(va_h, tr_h)
    nearest = dist.min(1)
    nearest_idx = dist.argmin(1)

    print("\n── 누수 지표 ──────────────────────────────")
    print(f"최근접 거리 중앙값 : {np.median(nearest):.0f}")
    print(f"최소 거리          : {nearest.min()}")
    for t in (5, 10, 20):
        cnt = int((nearest <= t).sum())
        print(f"  거리 <= {t:2d} : {cnt:3d}장 ({cnt / len(va_files) * 100:.1f}%)"
              + ("   ← near-duplicate" if t == 10 else ""))

    adj, rnd = baseline_distances(tr_files, tr_h)
    print("\n── 기준선 (이 데이터셋의 스케일) ──────────")
    print(f"인접 번호 프레임 쌍 평균 거리 : {adj:.1f}")
    print(f"무작위 프레임 쌍 평균 거리    : {rnd:.1f}")
    print("→ 최근접 거리 중앙값이 인접 쌍 평균에 가까우면 누수를 의심해야 합니다.")

    print("\n── 가장 의심스러운 10쌍 ───────────────────")
    for k in np.argsort(nearest)[:10]:
        print(f"  d={nearest[k]:3d}  val/{va_files[k].name:<14} <-> train/{tr_files[nearest_idx[k]].name}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
