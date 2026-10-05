"""Keep alive without a broker: a fake server speaks just enough MQTT 5 to show
what the client puts on the wire (CONNECT keep alive field, PINGREQ cadence)."""

from __future__ import annotations

import anyio
import pytest
from anyio.abc import SocketStream

from mqttproto.async_client import KEEP_ALIVE_PING_FRACTION, AsyncMQTTClient

pytestmark = pytest.mark.anyio

CONNACK = b"\x20\x03\x00\x00\x00"  # session present 0, SUCCESS, no properties
PINGREQ = b"\xc0\x00"
PINGRESP = b"\xd0\x00"
CONNECT_PACKET_TYPE = 0x10
# CONNECT variable header: 00 04 'MQTT' | version | flags | keep alive (2 bytes)
KEEP_ALIVE_OFFSET = 2 + 4 + 1 + 1


def _split_header(buf: bytes) -> tuple[int, int]:
    """Return (remaining length, header size) of the packet at the start of buf."""
    length = 0
    multiplier = 1
    pos = 1
    while True:
        byte = buf[pos]
        length += (byte & 0x7F) * multiplier
        pos += 1
        if not byte & 0x80:
            return length, pos
        multiplier *= 128


class FakeBroker:
    def __init__(self) -> None:
        self.connect_keep_alive: int | None = None
        self.ping_times: list[float] = []

    async def handle(self, stream: SocketStream) -> None:
        buf = b""
        async with stream:
            try:
                async for chunk in stream:
                    buf += chunk
                    while len(buf) >= 2:
                        length, header = _split_header(buf)
                        if len(buf) < header + length:
                            break
                        packet, buf = buf[: header + length], buf[header + length :]
                        await self._on_packet(stream, packet, header)
            except (anyio.EndOfStream, anyio.BrokenResourceError):
                pass

    async def _on_packet(
        self, stream: SocketStream, packet: bytes, header: int
    ) -> None:
        if packet[0] & 0xF0 == CONNECT_PACKET_TYPE:
            body = packet[header:]
            self.connect_keep_alive = int.from_bytes(
                body[KEEP_ALIVE_OFFSET : KEEP_ALIVE_OFFSET + 2], "big"
            )
            await stream.send(CONNACK)
        elif packet == PINGREQ:
            self.ping_times.append(anyio.current_time())
            await stream.send(PINGRESP)


async def _run_client(keep_alive: int | None, hold_s: float) -> FakeBroker:
    broker = FakeBroker()
    listener = await anyio.create_tcp_listener(local_host="127.0.0.1")
    port = listener.extra(anyio.abc.SocketAttribute.local_port)
    # stamina_kwargs: the fork's validator rejects its own None default (pre-existing).
    kwargs: dict = {"stamina_kwargs": {"attempts": 1}}
    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive
    async with anyio.create_task_group() as tg:
        tg.start_soon(listener.serve, broker.handle)
        async with AsyncMQTTClient("127.0.0.1", port, **kwargs):
            await anyio.sleep(hold_s)
        tg.cancel_scope.cancel()
    return broker


async def test_keep_alive_is_sent_in_connect_and_pings_go_out() -> None:
    keep_alive = 1
    hold_s = keep_alive * KEEP_ALIVE_PING_FRACTION * 3 + 0.3
    started = anyio.current_time()
    broker = await _run_client(keep_alive, hold_s)

    assert broker.connect_keep_alive == keep_alive
    assert len(broker.ping_times) >= 3
    # The broker drops a client silent for 1.5 x keep alive: no gap may come near it.
    gaps = [b - a for a, b in zip([started, *broker.ping_times], broker.ping_times)]
    assert max(gaps) < keep_alive


async def test_default_keep_alive_is_zero_and_sends_no_pings() -> None:
    broker = await _run_client(None, 1.0)

    assert broker.connect_keep_alive == 0
    assert broker.ping_times == []


async def test_keep_alive_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncMQTTClient(keep_alive=65536, stamina_kwargs={})


async def test_ping_loop_ends_with_the_connection_not_with_its_timer() -> None:
    """The ping loop shares the per-connection task group with the reader, so it must
    end as soon as the link is lost; with a 60 s keep alive it would otherwise hold
    the teardown of a dropped connection for up to 45 s."""
    client = AsyncMQTTClient(keep_alive=60, stamina_kwargs={})
    connection_lost = anyio.Event()
    with anyio.fail_after(1):
        async with anyio.create_task_group() as tg:
            tg.start_soon(client._send_pings, connection_lost)
            await anyio.sleep(0.05)
            connection_lost.set()
