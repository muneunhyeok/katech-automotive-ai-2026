"""
CAN 통신 기본 예제 — 수신 / 송신 / DBC 디코딩

셋업 순서 (이 순서를 지켜야 디버깅이 쉽습니다):
  1. 드라이버 설치 — Kvaser 사용 시 kvaser-linuxcan
  2. 비트레이트 정합 — 차량 버스에 맞춤 (보통 500kbps)
  3. 수신 확인 — candump 로 프레임이 실제로 들어오는지 먼저 검증

  ※ 비트레이트가 틀리면 아무것도 안 들어오거나 에러 프레임만 쌓입니다.
    코드 문제로 오해하기 쉬우니, 반드시 candump 로 먼저 확인하세요.

인터페이스 준비 (SocketCAN):
    sudo ip link set can0 type can bitrate 500000
    sudo ip link set up can0
    candump can0          # 여기서 프레임이 보여야 다음 단계로

가상 인터페이스로 테스트하려면:
    sudo modprobe vcan
    sudo ip link add dev vcan0 type vcan
    sudo ip link set up vcan0
"""

import can


CHANNEL = "can0"      # 가상 테스트 시 "vcan0"
BUSTYPE = "socketcan"  # Kvaser 전용 드라이버 사용 시 "kvaser"


def receive(duration_frames: int = 20):
    """버스에 흐르는 프레임을 읽어 ID와 페이로드를 출력"""
    bus = can.interface.Bus(channel=CHANNEL, bustype=BUSTYPE)
    print(f"{'ID':>8}  {'DLC':>3}  DATA")
    for _ in range(duration_frames):
        msg = bus.recv(timeout=1.0)
        if msg is None:
            print("timeout — 비트레이트와 배선을 확인하세요")
            break
        payload = " ".join(f"{b:02X}" for b in msg.data)
        print(f"{hex(msg.arbitration_id):>8}  {msg.dlc:>3}  {payload}")
    bus.shutdown()


def send():
    """단일 프레임 송신

    ID는 주소가 아니라 '메시지 종류'이며, 숫자가 작을수록 우선순위가 높습니다.
    여러 노드가 동시에 송신하면 ID 비교로 중재(arbitration)됩니다.
    """
    bus = can.interface.Bus(channel=CHANNEL, bustype=BUSTYPE)
    msg = can.Message(
        arbitration_id=0x123,
        data=[0x11, 0x22, 0x33, 0x44],
        is_extended_id=False,   # 11bit 표준 포맷 (확장은 29bit)
    )
    bus.send(msg)
    print(f"sent: {hex(msg.arbitration_id)} {list(msg.data)}")
    bus.shutdown()


def decode_with_dbc(dbc_path: str = "vehicle.dbc", n: int = 10):
    """DBC로 원시 바이트를 물리값으로 변환

    DBC 없이는 8바이트가 의미 없는 숫자입니다.
    DBC가 "몇 번째 비트가 무슨 신호이고 스케일이 얼마인지"를 정의합니다.
    """
    import cantools

    db = cantools.database.load_file(dbc_path)
    bus = can.interface.Bus(channel=CHANNEL, bustype=BUSTYPE)

    for _ in range(n):
        msg = bus.recv(timeout=1.0)
        if msg is None:
            break
        try:
            decoded = db.decode_message(msg.arbitration_id, msg.data)
            print(f"{hex(msg.arbitration_id)}  {decoded}")
        except KeyError:
            # DBC에 정의되지 않은 ID — 정상입니다. 차량은 많은 메시지를 흘립니다
            continue
    bus.shutdown()


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "recv"
    {"recv": receive, "send": send, "dbc": decode_with_dbc}[mode]()
