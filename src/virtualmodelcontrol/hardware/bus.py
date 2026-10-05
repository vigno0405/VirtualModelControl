"""The Dynamixel bus: the motors' control tables, read and written through the Dynamixel SDK
(protocol 2.0), or held in memory by ``FakeBus`` for tests and dry runs. Deprecated: leaves in
0.4.0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

# Control table of the X-series motors: address, size in bytes.
RETURN_DELAY_TIME = (9, 1)
OPERATING_MODE = (11, 1)
TORQUE_ENABLE = (64, 1)
STATUS_RETURN_LEVEL = (68, 1)
GOAL_CURRENT = (102, 2)
GOAL_VELOCITY = (104, 4)
PROFILE_VELOCITY = (112, 4)
GOAL_POSITION = (116, 4)
PRESENT_VELOCITY = (128, 4)
PRESENT_POSITION = (132, 4)
PRESENT_STATE = (128, 8)  # present velocity, then present position

OPERATING_MODES = {"torque": 0, "velocity": 1, "position": 4, "hold": 4}
"""Value of the operating mode register for each mode (4 is extended position control)."""

MODEL_NUMBERS = {1200: "XL330-M288", 1240: "XC330-T288", 1020: "XM430-W350"}
"""Model numbers a ping returns."""


def encode(value: int, size: int) -> bytes:
    """A signed integer as the bus's little-endian bytes."""
    return int(value).to_bytes(size, "little", signed=True)


def decode(data: bytes | bytearray) -> int:
    """Inverse of ``encode``."""
    return int.from_bytes(bytes(data), "little", signed=True)


class Bus(Protocol):
    """What a plant needs from a bus. Errors raise ``IOError``."""

    def ping(self, id: int) -> int | None:
        """The motor's model number, or None when it does not answer."""
        ...

    def ping_all(self) -> dict[int, int]:
        """Model number of every motor that answers, by ID."""
        ...

    def write(self, id: int, address: int, data: bytes) -> None:
        """Write bytes to one motor."""
        ...

    def read(self, id: int, address: int, size: int) -> bytes:
        """Read bytes from one motor."""
        ...

    def sync_read(self, ids: Sequence[int], address: int, size: int) -> dict[int, bytes]:
        """The same bytes of several motors, in one transaction."""
        ...

    def sync_write(self, address: int, data: Mapping[int, bytes]) -> None:
        """Bytes to several motors (same address and size), in one transaction."""
        ...

    def close(self) -> None:
        """Release the port."""
        ...


class SdkBus:
    """A serial port through the Dynamixel SDK (protocol 2.0)."""

    def __init__(self, port: str, baudrate: int) -> None:
        import dynamixel_sdk as sdk

        self.sdk = sdk
        self.port = sdk.PortHandler(port)
        self.packet = sdk.PacketHandler(2.0)
        if not self.port.openPort():
            raise OSError(f"cannot open {port}")
        if not self.port.setBaudRate(baudrate):
            self.port.closePort()
            raise OSError(f"cannot set {port} to {baudrate} bit/s")

    def _check(self, result: int, error: int, what: str) -> None:
        if result != self.sdk.COMM_SUCCESS:
            raise OSError(f"{what}: {self.packet.getTxRxResult(result)}")
        if error:
            raise OSError(f"{what}: {self.packet.getRxPacketError(error)}")

    def ping(self, id: int) -> int | None:
        """The motor's model number, or None when it does not answer."""
        model, result, _ = self.packet.ping(self.port, id)
        return int(model) if result == self.sdk.COMM_SUCCESS else None

    def ping_all(self) -> dict[int, int]:
        """Model number of every motor that answers, by ID (one broadcast ping)."""
        found, result = self.packet.broadcastPing(self.port)
        if result != self.sdk.COMM_SUCCESS:
            return {}
        return {int(i): int(info[0]) for i, info in found.items()}

    def write(self, id: int, address: int, data: bytes) -> None:
        """Write bytes to one motor."""
        result, error = self.packet.writeTxRx(self.port, id, address, len(data), list(data))
        self._check(result, error, f"motor {id}, write {address}")

    def read(self, id: int, address: int, size: int) -> bytes:
        """Read bytes from one motor."""
        data, result, error = self.packet.readTxRx(self.port, id, address, size)
        self._check(result, error, f"motor {id}, read {address}")
        return bytes(data)

    def sync_read(self, ids: Sequence[int], address: int, size: int) -> dict[int, bytes]:
        """The same bytes of several motors, in one transaction."""
        group = self.sdk.GroupSyncRead(self.port, self.packet, address, size)
        for i in ids:
            group.addParam(i)
        self._check(group.txRxPacket(), 0, f"sync read {address}")
        return {i: bytes(group.data_dict[i]) for i in ids}

    def sync_write(self, address: int, data: Mapping[int, bytes]) -> None:
        """Bytes to several motors (same address and size), in one transaction."""
        size = len(next(iter(data.values())))
        group = self.sdk.GroupSyncWrite(self.port, self.packet, address, size)
        for i, d in data.items():
            group.addParam(i, list(d))
        self._check(group.txPacket(), 0, f"sync write {address}")

    def close(self) -> None:
        """Release the port."""
        self.port.closePort()


class FakeBus:
    """Motors held in memory, for tests and dry runs: ``motors`` maps each ID to its model.

    Writes land in the control tables; a goal position written with the torque on in position
    mode moves the motor there at once. ``set_position`` and ``set_velocity`` play the robot.
    """

    def __init__(self, motors: Mapping[int, str], ticks: Mapping[int, int] | None = None) -> None:
        numbers = {name: number for number, name in MODEL_NUMBERS.items()}
        self.models = {i: numbers[m] for i, m in motors.items()}
        self.tables = {i: bytearray(256) for i in motors}
        self.log: list[tuple[str, Any]] = []  # every write, in order
        self.closed = False
        for i, t in (ticks or {}).items():
            self.set_position(i, t)

    def _table(self, id: int) -> bytearray:
        if id not in self.tables:
            raise OSError(f"motor {id} does not answer")
        return self.tables[id]

    def value(self, id: int, field: tuple[int, int]) -> int:
        """A register's value, as an integer."""
        address, size = field
        return decode(self._table(id)[address : address + size])

    def set_position(self, id: int, ticks: int) -> None:
        """Move a motor (present position, ticks)."""
        self._table(id)[132:136] = encode(ticks, 4)

    def set_velocity(self, id: int, raw: int) -> None:
        """Set a motor's present velocity (raw units)."""
        self._table(id)[128:132] = encode(raw, 4)

    def ping(self, id: int) -> int | None:
        """The motor's model number, or None when it does not answer."""
        return self.models.get(id)

    def ping_all(self) -> dict[int, int]:
        """Model number of every motor that answers, by ID."""
        return dict(self.models)

    def write(self, id: int, address: int, data: bytes) -> None:
        """Write bytes to one motor."""
        table = self._table(id)
        table[address : address + len(data)] = data
        self.log.append(("write", (id, address, decode(data))))
        torque_on = table[TORQUE_ENABLE[0]] == 1
        if address == GOAL_POSITION[0] and torque_on and table[OPERATING_MODE[0]] == 4:
            table[132:136] = data

    def read(self, id: int, address: int, size: int) -> bytes:
        """Read bytes from one motor."""
        return bytes(self._table(id)[address : address + size])

    def sync_read(self, ids: Sequence[int], address: int, size: int) -> dict[int, bytes]:
        """The same bytes of several motors, in one transaction."""
        return {i: self.read(i, address, size) for i in ids}

    def sync_write(self, address: int, data: Mapping[int, bytes]) -> None:
        """Bytes to several motors (same address and size), in one transaction."""
        for i, d in data.items():
            self.write(i, address, d)

    def close(self) -> None:
        """Release the port."""
        self.closed = True
