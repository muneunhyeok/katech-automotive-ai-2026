"""
시퀀스 단위 데이터 재분할 — 인접 프레임 누수 차단

무작위 분할이 주행 영상 데이터셋에서 위험한 이유:
연속 촬영된 프레임은 이웃끼리 거의 같은 장면이므로, 무작위로 나누면
train과 val에 사실상 동일한 이미지가 나뉘어 들어갑니다.

해법:
  1. BLOCK 프레임을 한 덩어리로 묶어 통째로 배정 (덩어리가 쪼개지지 않음)
  2. 덩어리 경계 앞뒤 BUFFER 프레임은 버림 (경계면 누수 차단)

결과 (docs/05-data-integrity-audit.md):
  거리<=10 near-duplicate 17% → 0%,  최근접 거리 중앙값 18 → 34

사용법:
    python split_sequence.py --src data --dst data_seq
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg")


def frame_index(p: Path) -> int:
    """파일명에서 프레임 번호 추출 (예: 0123.png -> 123)"""
    digits = "".join(c for c in p.stem if c.isdigit())
    return int(digits) if digits else -1


def collect(src: Path):
    """src 아래 모든 이미지를 프레임 번호 순으로 수집"""
    imgs = []
    for d in (src / "images").rglob("*"):
        if d.suffix.lower() in IMG_EXTS:
            imgs.append(d)
    imgs = [p for p in imgs if frame_index(p) >= 0]
    return sorted(imgs, key=frame_index)


def make_blocks(imgs, block: int, buffer: int):
    """BLOCK 단위로 덩어리를 만들고, 각 덩어리 양끝 BUFFER 프레임을 제거"""
    blocks = []
    for i in range(0, len(imgs), block):
        chunk = imgs[i:i + block]
        if len(chunk) <= 2 * buffer:
            continue
        blocks.append(chunk[buffer:len(chunk) - buffer])
    return blocks


def label_for(img: Path, src: Path) -> Path:
    """이미지 경로에 대응하는 라벨 경로"""
    rel = img.relative_to(src / "images")
    return (src / "labels" / rel).with_suffix(".txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data")
    ap.add_argument("--dst", default="data_seq")
    ap.add_argument("--block", type=int, default=60)
    ap.add_argument("--buffer", type=int, default=10)
    ap.add_argument("--ratio", default="0.75,0.15,0.10",
                    help="train,val,test 비율")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    random.seed(args.seed)

    imgs = collect(src)
    blocks = make_blocks(imgs, args.block, args.buffer)
    random.shuffle(blocks)

    r_tr, r_va, r_te = (float(x) for x in args.ratio.split(","))
    n = len(blocks)
    n_tr = int(n * r_tr)
    n_va = int(n * r_va)
    parts = {
        "train": blocks[:n_tr],
        "val":   blocks[n_tr:n_tr + n_va],
        "test":  blocks[n_tr + n_va:],
    }

    missing_labels = []
    for split, bs in parts.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        count = 0
        for b in bs:
            for img in b:
                shutil.copy(img, dst / "images" / split / img.name)
                lb = label_for(img, src)
                if lb.exists():
                    shutil.copy(lb, dst / "labels" / split / lb.name)
                else:
                    missing_labels.append(img.name)
                count += 1
        print(f"{split:5s}: {len(bs):3d} blocks / {count:4d} images")

    if missing_labels:
        print(f"\n[경고] 라벨 파일이 없는 이미지 {len(missing_labels)}장: {missing_labels[:10]}")
        print("  → YOLO는 이를 배경 이미지로 취급합니다. 실제 객체가 있다면 라벨링 누락입니다.")

    yaml = (dst / "data.yaml")
    yaml.write_text(
        f"path: {dst.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 3\n"
        "names:\n"
        "  0: vehicle\n"
        "  1: pedestrian\n"
        "  2: cycle\n",
        encoding="utf-8",
    )
    print(f"\n생성 완료: {yaml}")
    print("→ audit_leakage.py 로 재검증하세요:")
    print(f"   python audit_leakage.py {dst}/images/train {dst}/images/val")


if __name__ == "__main__":
    main()
