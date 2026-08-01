# 07. CAN 통신 및 실차 데이터 수집

## CAN이란

차량 안의 수십 개 ECU(엔진, 브레이크, 계기판, 조향 등)가 **두 가닥의 선(CAN High / CAN Low)을 공유하며** 메시지를 주고받는 차량 표준 네트워크입니다.

일반적인 네트워크와 다른 점 세 가지가 핵심입니다.

| 특징 | 내용 |
|---|---|
| **중앙 서버 없음** | 모든 노드가 같은 버스에 방송(broadcast). 마스터가 없습니다 |
| **주소가 아니라 ID** | 메시지에 수신자 주소 대신 ID가 붙습니다. "누구에게"가 아니라 "무엇에 관한 메시지인가" |
| **ID로 우선순위 결정** | **ID 숫자가 작을수록 우선순위가 높습니다.** 여러 노드가 동시에 송신하면 ID 비교로 중재(arbitration) |

브레이크 관련 메시지에 낮은 ID를 할당하면 버스가 혼잡해도 먼저 전달된다는 설계 — 안전 시스템에서 이 구조가 왜 중요한지가 바로 이해되는 부분입니다.

## CAN 프레임 구조

```
┌──────────┬─────┬──────────────────┬──────────┐
│ ID(11bit)│ DLC │ DATA (최대 8byte)│ CRC 등   │
└──────────┴─────┴──────────────────┴──────────┘
```

- **ID** — 메시지 식별자 겸 우선순위 (확장 포맷은 29bit)
- **DLC** — 데이터 길이 (Data Length Code)
- **DATA** — 실제 페이로드, 최대 8바이트

**8바이트 안에 RPM·속도·조향각이 어떤 위치에 어떤 스케일로 들어있는지**를 정의한 문서가 **DBC 파일**입니다. DBC 없이는 원시 바이트가 의미 없는 숫자일 뿐입니다.

```
raw CAN frame  ──[DBC로 디코딩]──▶  RPM 2350, 속도 62km/h, 조향각 -3.2°
```

실차 데이터 수집이란 결국 **버스에 흐르는 메시지를 로깅하고, DBC로 디코딩해 물리값으로 변환하는 작업**입니다.

---

## 실습 장비 — Kvaser USB-CAN

<p align="center">
  <img src="../assets/screenshots/can-kvaser.jpg" width="440" alt="Kvaser USB-CAN 인터페이스">
</p>

PC와 CAN 버스를 잇는 USB 인터페이스로 Kvaser를 사용했습니다. Ubuntu 환경에서 진행했고, 셋업 순서는 세 단계입니다.

1. **드라이버 설치** — Kvaser Linux 드라이버(kvaser-linuxcan)
2. **비트레이트 정합** — 차량 버스에 맞춤 (일반적으로 500kbps)
3. **수신 확인** — 프레임이 실제로 들어오는지 먼저 검증

이 순서를 지키는 게 중요합니다. **비트레이트가 틀리면 아무것도 안 들어오거나 에러 프레임만 쌓이는데**, 코드 문제로 오해하기 쉽습니다.

## python-can 기본 사용

```python
import can

bus = can.interface.Bus(channel="can0", bustype="socketcan")

# 수신
msg = bus.recv(timeout=1.0)
print(hex(msg.arbitration_id), msg.data)

# 송신
tx = can.Message(arbitration_id=0x123,
                 data=[0x11, 0x22],
                 is_extended_id=False)
bus.send(tx)
```

### DBC 디코딩 (cantools)

```python
import cantools

db = cantools.database.load_file("vehicle.dbc")
decoded = db.decode_message(msg.arbitration_id, msg.data)
print(decoded)      # {'EngineRPM': 2350.0, 'VehicleSpeed': 62.0, ...}
```

### 알아둘 도구

| 도구 | 용도 |
|---|---|
| `SocketCAN` | 리눅스 커널의 CAN 네트워크 스택. 인터페이스를 네트워크 장치처럼 다룸 |
| `candump` | 터미널에서 실시간 프레임 확인 — 디버깅 1순위 |
| `cantools` | DBC 파싱 및 디코딩 |
| `OBD-II` | 진단 커넥터 표준 (차량 진단 포트) |

전체 예제는 [`src/can_example.py`](../src/can_example.py)에 있습니다.

---

## 실차 데이터 수집

자율주행 시험운행 차량(현대 IONIQ 6, 루프탑 센서 마운트)에서 실제 주행 데이터를 수집하는 과정을 참관·실습했습니다.

여기서 확인한 것이 앞선 객체 검출 실습과 이어집니다.

> **모델이 다루는 650장의 이미지는 이런 차량이 실제 도로를 달리며 수집한 영상에서 나온 것이다.**

이 연결이 중요한 이유는, [05번 문서](05-data-integrity-audit.md)에서 다룬 **인접 프레임 누수 문제의 물리적 원인**이 바로 여기 있기 때문입니다. 주행 영상은 초당 수십 프레임으로 연속 촬영되므로 이웃한 프레임은 거의 같은 장면일 수밖에 없습니다. 이 사실을 알고 나면 "무작위 분할은 주행 데이터에서 위험하다"는 판단이 자연스럽게 따라옵니다.

---

## CAN과 인지 파이프라인의 관계

자율주행 3단계 파이프라인에서 CAN이 어디에 있는지 정리하면 이렇습니다.

```
인지(Perception)  →  판단(Planning)  →  제어(Control)
   카메라·라이다·레이더      경로·행동 결정        조향·가감속
        │                                          │
        └──────────── CAN 버스 ─────────────────────┘
              (센서 데이터 · 제어 명령 전달)
```

카메라로 사람을 검출하는 것(인지)과 그 결과로 브레이크를 거는 것(제어) 사이를 실제로 잇는 것이 CAN입니다. **AI 모델만 잘 만들어서는 차가 서지 않는다** — 이 실습에서 얻은 감각입니다.

## 관련 문서

- [03. 객체 검출](03-object-detection.md)
- [05. 데이터 무결성 감사](05-data-integrity-audit.md)
