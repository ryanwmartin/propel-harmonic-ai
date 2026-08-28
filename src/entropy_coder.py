"""Lossless adaptive arithmetic coding for binary neural-network state.

The coder treats serialized tensor bytes as symbols from a 256-value alphabet.
Both encoder and decoder start from the same uniform frequency model and update
it after every symbol, so no model-specific probability table is stored. The
format preserves the source bytes exactly and includes a CRC32 integrity check.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

ENTROPY_MAGIC = b"AEC1"
ENTROPY_VERSION = 1
_HEADER = struct.Struct("<4sBQI")
_STATE_BITS = 32
_FULL_RANGE = 1 << _STATE_BITS
_HALF_RANGE = _FULL_RANGE >> 1
_QUARTER_RANGE = _HALF_RANGE >> 1
_THREE_QUARTER_RANGE = _QUARTER_RANGE * 3
_MAXIMUM_TOTAL_FREQUENCY = 1 << 15
_SYMBOL_COUNT = 256


@dataclass(frozen=True)
class EntropyCodingMetrics:
    """Storage measurements for one completed arithmetic-code stream."""

    source_bytes: int
    encoded_bytes: int

    @property
    def encoded_to_source_ratio(self) -> float:
        return self.encoded_bytes / self.source_bytes if self.source_bytes else 0.0

    @property
    def bits_per_source_byte(self) -> float:
        return self.encoded_bytes * 8 / self.source_bytes if self.source_bytes else 0.0

    @property
    def bits_per_fp16_weight(self) -> float:
        if self.source_bytes % 2:
            raise ValueError("FP16 metrics require an even source byte count")
        weight_count = self.source_bytes // 2
        return self.encoded_bytes * 8 / weight_count if weight_count else 0.0


class _BitWriter:
    def __init__(self) -> None:
        self.output = bytearray()
        self.current_byte = 0
        self.current_bit_count = 0

    def write(self, bit: int) -> None:
        self.current_byte = (self.current_byte << 1) | bit
        self.current_bit_count += 1
        if self.current_bit_count == 8:
            self.output.append(self.current_byte)
            self.current_byte = 0
            self.current_bit_count = 0

    def finish(self) -> bytes:
        if self.current_bit_count:
            self.output.append(self.current_byte << (8 - self.current_bit_count))
        return bytes(self.output)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.bit_index = 0

    def read(self) -> int:
        if self.bit_index >= len(self.payload) * 8:
            # Arithmetic coding convention: the finite code is followed by an
            # unlimited zero tail. The declared length and CRC guard framing.
            return 0
        byte_value = self.payload[self.bit_index // 8]
        bit = (byte_value >> (7 - self.bit_index % 8)) & 1
        self.bit_index += 1
        return bit


class _AdaptiveFrequencyModel:
    """Adaptive byte frequencies backed by a Fenwick prefix-sum tree."""

    def __init__(self) -> None:
        self.frequencies = [1] * _SYMBOL_COUNT
        self.tree = [0] * (_SYMBOL_COUNT + 1)
        self.total = 0
        self._rebuild()

    def _rebuild(self) -> None:
        self.tree = [0] * (_SYMBOL_COUNT + 1)
        self.total = 0
        for symbol, frequency in enumerate(self.frequencies):
            self._add(symbol, frequency)
            self.total += frequency

    def _add(self, symbol: int, amount: int) -> None:
        tree_index = symbol + 1
        while tree_index <= _SYMBOL_COUNT:
            self.tree[tree_index] += amount
            tree_index += tree_index & -tree_index

    def cumulative(self, symbol: int) -> int:
        cumulative_frequency = 0
        tree_index = symbol
        while tree_index:
            cumulative_frequency += self.tree[tree_index]
            tree_index -= tree_index & -tree_index
        return cumulative_frequency

    def interval(self, symbol: int) -> tuple[int, int, int]:
        lower_frequency = self.cumulative(symbol)
        return lower_frequency, lower_frequency + self.frequencies[symbol], self.total

    def symbol_for_cumulative(self, target: int) -> int:
        if target < 0 or target >= self.total:
            raise ValueError("arithmetic-code cumulative frequency is out of range")
        symbol = 0
        prefix_frequency = 0
        search_step = 1 << (_SYMBOL_COUNT.bit_length() - 1)
        while search_step:
            candidate = symbol + search_step
            if candidate <= _SYMBOL_COUNT and prefix_frequency + self.tree[candidate] <= target:
                symbol = candidate
                prefix_frequency += self.tree[candidate]
            search_step >>= 1
        if symbol >= _SYMBOL_COUNT:
            raise ValueError("arithmetic-code symbol lookup failed")
        return symbol

    def update(self, symbol: int) -> None:
        self.frequencies[symbol] += 1
        self._add(symbol, 1)
        self.total += 1
        if self.total >= _MAXIMUM_TOTAL_FREQUENCY:
            self.frequencies = [(frequency + 1) // 2 for frequency in self.frequencies]
            self._rebuild()


def encode_bytes(source: bytes) -> bytes:
    """Return a self-contained, lossless arithmetic encoding of ``source``."""
    source = bytes(source)
    model = _AdaptiveFrequencyModel()
    bit_writer = _BitWriter()
    lower_bound = 0
    upper_bound = _FULL_RANGE - 1
    pending_underflow_bits = 0

    def emit_with_pending(bit: int) -> None:
        nonlocal pending_underflow_bits
        bit_writer.write(bit)
        for _ in range(pending_underflow_bits):
            bit_writer.write(1 - bit)
        pending_underflow_bits = 0

    for symbol in source:
        lower_frequency, upper_frequency, total_frequency = model.interval(symbol)
        interval_width = upper_bound - lower_bound + 1
        upper_bound = lower_bound + interval_width * upper_frequency // total_frequency - 1
        lower_bound = lower_bound + interval_width * lower_frequency // total_frequency

        while True:
            if upper_bound < _HALF_RANGE:
                emit_with_pending(0)
            elif lower_bound >= _HALF_RANGE:
                emit_with_pending(1)
                lower_bound -= _HALF_RANGE
                upper_bound -= _HALF_RANGE
            elif lower_bound >= _QUARTER_RANGE and upper_bound < _THREE_QUARTER_RANGE:
                pending_underflow_bits += 1
                lower_bound -= _QUARTER_RANGE
                upper_bound -= _QUARTER_RANGE
            else:
                break
            lower_bound = lower_bound << 1
            upper_bound = (upper_bound << 1) | 1

        model.update(symbol)

    pending_underflow_bits += 1
    emit_with_pending(0 if lower_bound < _QUARTER_RANGE else 1)
    payload = bit_writer.finish()
    checksum = zlib.crc32(source) & 0xFFFFFFFF
    return _HEADER.pack(
        ENTROPY_MAGIC, ENTROPY_VERSION, len(source), checksum
    ) + payload


def decode_bytes(encoded: bytes) -> bytes:
    """Decode an arithmetic stream and verify its length and CRC32 checksum."""
    if len(encoded) < _HEADER.size:
        raise ValueError("truncated entropy-code header")
    magic, version, source_length, expected_checksum = _HEADER.unpack_from(encoded)
    if magic != ENTROPY_MAGIC:
        raise ValueError("invalid entropy-code magic")
    if version != ENTROPY_VERSION:
        raise ValueError(f"unsupported entropy-code version {version}")
    payload = encoded[_HEADER.size :]
    if source_length and not payload:
        raise ValueError("truncated entropy-code payload")

    model = _AdaptiveFrequencyModel()
    bit_reader = _BitReader(payload)
    lower_bound = 0
    upper_bound = _FULL_RANGE - 1
    code_value = 0
    for _ in range(_STATE_BITS):
        code_value = (code_value << 1) | bit_reader.read()

    output = bytearray()
    for _ in range(source_length):
        interval_width = upper_bound - lower_bound + 1
        target = ((code_value - lower_bound + 1) * model.total - 1) // interval_width
        symbol = model.symbol_for_cumulative(target)
        lower_frequency, upper_frequency, total_frequency = model.interval(symbol)
        upper_bound = lower_bound + interval_width * upper_frequency // total_frequency - 1
        lower_bound = lower_bound + interval_width * lower_frequency // total_frequency

        while True:
            if upper_bound < _HALF_RANGE:
                pass
            elif lower_bound >= _HALF_RANGE:
                lower_bound -= _HALF_RANGE
                upper_bound -= _HALF_RANGE
                code_value -= _HALF_RANGE
            elif lower_bound >= _QUARTER_RANGE and upper_bound < _THREE_QUARTER_RANGE:
                lower_bound -= _QUARTER_RANGE
                upper_bound -= _QUARTER_RANGE
                code_value -= _QUARTER_RANGE
            else:
                break
            lower_bound = lower_bound << 1
            upper_bound = (upper_bound << 1) | 1
            code_value = (code_value << 1) | bit_reader.read()

        output.append(symbol)
        model.update(symbol)

    result = bytes(output)
    actual_checksum = zlib.crc32(result) & 0xFFFFFFFF
    if actual_checksum != expected_checksum:
        raise ValueError("entropy-code checksum mismatch")
    return result


def measure_encoding(source: bytes, encoded: bytes) -> EntropyCodingMetrics:
    """Build storage metrics after validating a completed round trip."""
    if decode_bytes(encoded) != bytes(source):
        raise ValueError("encoded stream does not reproduce the source exactly")
    return EntropyCodingMetrics(len(source), len(encoded))
