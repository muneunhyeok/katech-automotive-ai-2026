# src — 코드

실습에서 사용한 스크립트를 재현 가능한 형태로 재구성한 것입니다.
데이터셋과 가중치는 주관 기관 자산이라 포함하지 않았습니다.

| 파일 | 용도 | 관련 문서 |
|---|---|---|
| [`analyze_dataset.py`](analyze_dataset.py) | 클래스별 객체 크기 분포 측정 — **튜닝 전 병목 진단** | [03](../docs/03-object-detection.md) |
| [`audit_leakage.py`](audit_leakage.py) | perceptual hash 기반 train/val 프레임 중복 정량화 | [05](../docs/05-data-integrity-audit.md) |
| [`split_sequence.py`](split_sequence.py) | 시퀀스 단위 재분할 (BLOCK/BUFFER) — 누수 교정 | [05](../docs/05-data-integrity-audit.md) |
| [`inspect_checkpoint.py`](inspect_checkpoint.py) | 사전학습 가중치가 정말 COCO인지 메타데이터 검사 | [05](../docs/05-data-integrity-audit.md) |
| [`train.py`](train.py) | YOLO11 최종 학습 설정 | [03](../docs/03-object-detection.md), [04](../docs/04-experiment-log.md) |
| [`evaluate.py`](evaluate.py) | 여러 split 교차 평가 — 누수 프리미엄 측정 | [04](../docs/04-experiment-log.md) |
| [`can_example.py`](can_example.py) | CAN 수신·송신·DBC 디코딩 | [07](../docs/07-can-communication.md) |

## 권장 실행 순서

```bash
# 1. 병목이 무엇인지 먼저 측정
python analyze_dataset.py data/images/train data/labels/train

# 2. 데이터 누수 감사 — 이걸 첫날에 했어야 했습니다
python audit_leakage.py data/images/train data/images/valid

# 3. 누수가 있으면 시퀀스 단위로 재분할
python split_sequence.py --src data --dst data_seq --block 60 --buffer 10
python audit_leakage.py data_seq/images/train data_seq/images/val   # 재검증

# 4. 제공받은 가중치가 정말 COCO인지 확인
python inspect_checkpoint.py pre_yolo11n.pt --zero-step data_seq/data.yaml

# 5. 학습 (백그라운드 — 세션이 끊겨도 유지)
nohup python -u train.py > train.log 2>&1 &
tail -f train.log

# 6. 교차 평가
python evaluate.py clean_final.pt
```

## 의존성

```bash
pip install ultralytics numpy pillow
pip install "lap>=0.5.12"        # 다중 객체 추적용
pip install python-can cantools  # CAN 실습용
```
