"""
체크포인트 메타데이터 검사 — "이 가중치가 정말 COCO 사전학습인가?"

실습에서 제공받은 pre_yolo11n.pt 를 COCO 가중치로 알고 썼는데,
실제로는 같은 데이터셋에 이미 fine-tuning된 모델이었습니다.
이걸 모르고 튜닝하면 "성능을 올린 것"이 아니라
"완성된 모델을 얼마나 덜 망가뜨렸는가"를 측정하게 됩니다.

★ import ultralytics 를 torch.load 보다 먼저 해야 합니다.
  그러지 않으면 커스텀 클래스 unpickling에서 실패합니다.

사용법:
    python inspect_checkpoint.py pre_yolo11n.pt
    python inspect_checkpoint.py pre_yolo11n.pt --zero-step data/data.yaml
"""

import argparse

import ultralytics  # noqa: F401  ★ torch.load 보다 먼저
import torch


def inspect(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    names = getattr(ckpt.get("model"), "names", None)
    print(f"\n파일: {path}")
    print(f"클래스 수 : {len(names) if names else '?'}")
    print(f"클래스명  : {names}")

    if names and len(names) == 80:
        print("  → 80클래스 = 순수 COCO 가중치일 가능성이 높습니다.")
    elif names:
        print("  → ⚠ 커스텀 클래스명입니다. 이미 fine-tuning된 가중치입니다.")

    for key in ("epoch", "best_fitness", "date", "version"):
        if key in ckpt:
            print(f"{key:<10}: {ckpt[key]}")

    ta = ckpt.get("train_args")
    if ta:
        print("\n[train_args] — 이 가중치가 어떻게 만들어졌는지")
        for k in ("model", "data", "epochs", "imgsz", "optimizer"):
            if k in ta:
                print(f"  {k:<10}: {ta[k]}")


def zero_step_check(path: str, data_yaml: str, imgsz: int = 1536):
    """학습 0스텝 상태의 성능 — 가장 결정적인 증거"""
    from ultralytics import YOLO
    r = YOLO(path).val(data=data_yaml, imgsz=imgsz, verbose=False, plots=False)
    print(f"\n[0스텝 성능] {path}  mAP50 = {r.box.map50:.4f}")
    print("  순수 COCO 가중치라면 커스텀 클래스에서 0에 가까워야 정상입니다.")
    print("  (실측: yolo11n.pt = 0.0009 vs pre_yolo11n.pt = 0.7638)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--zero-step", metavar="DATA_YAML", default=None)
    ap.add_argument("--imgsz", type=int, default=1536)
    a = ap.parse_args()

    inspect(a.weights)
    if a.zero_step:
        zero_step_check(a.weights, a.zero_step, a.imgsz)
