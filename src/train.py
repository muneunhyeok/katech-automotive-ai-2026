"""
YOLO11 학습 스크립트 — KATECH 실습 최종 설정

설계 근거는 docs/03-object-detection.md, docs/04-experiment-log.md 참고.
핵심 원칙 세 가지가 코드에 반영되어 있습니다.

  1. IMGSZ를 상수로 두고 학습·검증에서 동일하게 사용 (해상도 불일치로 인한 허수 방지)
  2. best.pt 경로를 문자열로 조립하지 않고 model.trainer.best를 사용 (경로 중복 버그 회피)
  3. last.pt가 있으면 resume — 플랫폼 GPU 재할당으로 인한 학습 중단에 대비
"""

import os

# ── MIG 환경 대응 ───────────────────────────────────────────────
# NVML이 보고하는 MIG 인스턴스와 CUDA가 노출하는 디바이스가 불일치할 수 있음.
# 반드시 torch import 이전에 설정해야 적용됩니다.
# os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

from pathlib import Path
import shutil
from ultralytics import YOLO

# ── 설정 ────────────────────────────────────────────────────────
IMGSZ = 1536                      # 학습·검증·추론에서 동일하게 사용할 것
DATA = Path("data_seq/data.yaml").resolve()   # 누수 교정된 split
START = "yolo11s.pt"              # 순수 COCO 가중치에서 시작 (오염 소스 차단)
PROJECT = "runs"                  # "runs/detect" 로 주면 경로가 이중으로 붙음
NAME = "clean_s_1536"

RUN = Path(PROJECT) / NAME
LAST = RUN / "weights" / "last.pt"


def build_args():
    return dict(
        data=str(DATA),
        epochs=150,
        patience=0,                # close_mosaic 구간이 잘리지 않도록 조기종료 비활성
        imgsz=IMGSZ,
        batch=12,
        device=0,
        workers=2,
        amp=True,
        cache="ram",

        # 옵티마이저 — 소규모 데이터셋에서 SGD 대비 수렴 안정
        optimizer="AdamW",
        lr0=0.0007,
        lrf=0.05,
        cos_lr=True,
        weight_decay=0.0075,       # 0.001 → 0.0075, +0.0160 mAP50-95 (노이즈의 1.8배)
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.02,

        # 손실 가중치
        box=9.0,                   # 박스 회귀 가중치 상향 → precision 개선
        cls_pw=0.25,               # 클래스 불균형(vehicle 편중) 대응
        dfl=1.0,                   # 명시 설정만으로 유의미한 상승

        # augmentation
        mosaic=1.0,
        close_mosaic=25,           # 후반 25 epoch 동안 mosaic 종료 → 계단식 상승 구간
        degrees=5.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.0,                # 주행 영상은 상하 반전이 물리적으로 무의미

        project=PROJECT,
        name=NAME,
        exist_ok=True,
        seed=42,
        deterministic=True,
    )


def main():
    if LAST.exists():
        print(f"[resume] {LAST}")
        model = YOLO(str(LAST))
        model.train(resume=True)
    else:
        model = YOLO(START)
        model.train(**build_args())

    # ★ 경로를 수동으로 조립하지 말 것 — 학습 객체가 알려주는 실제 경로를 사용
    best = Path(model.trainer.best)
    shutil.copy(best, "clean_final.pt")
    print(f"[saved] {best} -> clean_final.pt")

    # 최종 평가: val(모델 선택용) / test(보고용) 분리
    final = YOLO(str(best))
    for split in ("val", "test"):
        r = final.val(data=str(DATA), split=split, imgsz=IMGSZ,
                      batch=8, device=0, conf=0.001, iou=0.7,
                      plots=False, verbose=False)
        per_class = " ".join(f"{n}:{r.box.maps[i]:.3f}" for i, n in r.names.items())
        print(f"[{split:5s}] mAP50 {r.box.map50:.4f} | mAP50-95 {r.box.map:.4f} | {per_class}")


if __name__ == "__main__":
    main()
