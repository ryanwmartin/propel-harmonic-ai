"""Exact search compressors for decimal programs and arbitrary binary rows.

This module deliberately chooses among a small set of fully self-describing,
lossless representations. Candidate sizes are their actual serialized byte
lengths; a representation is accepted only after decoding and exact comparison.
The shared decoder implementation itself is not part of each payload.
"""

from __future__ import annotations

import lzma
import math
import zlib
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from typing import Callable, Iterable

MAGIC = b"LSC1"


class EncodingKind(IntEnum):
    RAW_BYTES = 0
    PERIODIC_BYTES = 1
    ZLIB_BYTES = 2
    LZMA_BYTES = 3
    RAW_DECIMAL = 16
    PERIODIC_DECIMAL = 17
    RATIONAL_DECIMAL = 18
    MODULAR_DECIMAL = 19
    POWER_OFFSET_DECIMAL = 20


@dataclass(frozen=True)
class SearchEncoding:
    """A serialized, self-describing result from the exact search."""

    kind: EncodingKind
    original_length: int
    payload: bytes

    def serialize(self) -> bytes:
        return MAGIC + bytes((int(self.kind),)) + _encode_varint(self.original_length) + self.payload

    @property
    def encoded_bytes(self) -> int:
        return len(self.serialize())

    @property
    def encoded_bits(self) -> int:
        return self.encoded_bytes * 8


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varints cannot encode negative values")
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        result.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(result)


def _decode_varint(data: bytes, offset: int = 0) -> tuple[int, int]:
    value = 0
    shift = 0
    for index in range(offset, len(data)):
        byte = data[index]
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index + 1
        shift += 7
        if shift > 63:
            raise ValueError("varint exceeds 64 bits")
    raise ValueError("truncated varint")


def _unsigned_bytes(value: int) -> bytes:
    if value < 0:
        raise ValueError("expected an unsigned integer")
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")


def _pack_integer(value: int) -> bytes:
    encoded = _unsigned_bytes(value)
    return _encode_varint(len(encoded)) + encoded


def _unpack_integer(payload: bytes, offset: int) -> tuple[int, int]:
    byte_count, offset = _decode_varint(payload, offset)
    end = offset + byte_count
    if byte_count < 1 or end > len(payload):
        raise ValueError("truncated integer")
    return int.from_bytes(payload[offset:end], "big"), end


def parse_encoding(serialized: bytes) -> SearchEncoding:
    """Parse a serialized search encoding with strict framing validation."""
    if not serialized.startswith(MAGIC) or len(serialized) <= len(MAGIC):
        raise ValueError("invalid search-compressor magic or truncated header")
    try:
        kind = EncodingKind(serialized[len(MAGIC)])
    except ValueError as error:
        raise ValueError("unknown search-compressor encoding kind") from error
    original_length, payload_offset = _decode_varint(serialized, len(MAGIC) + 1)
    return SearchEncoding(kind, original_length, serialized[payload_offset:])


def _smallest_period(data: bytes) -> bytes | None:
    """Return the shortest exact repeating prefix, including a partial final repeat."""
    if len(data) < 2:
        return None
    # Prefix-function gives the only minimal-period candidate in linear time.
    prefix = [0] * len(data)
    for index in range(1, len(data)):
        matched = prefix[index - 1]
        while matched and data[index] != data[matched]:
            matched = prefix[matched - 1]
        if data[index] == data[matched]:
            matched += 1
        prefix[index] = matched
    period_length = len(data) - prefix[-1]
    if period_length == len(data):
        return None
    period = data[:period_length]
    if all(value == period[index % period_length] for index, value in enumerate(data)):
        return period
    return None


def _verified_best(
    source: bytes | str,
    candidates: Iterable[SearchEncoding],
    decoder: Callable[[SearchEncoding], bytes | str],
) -> SearchEncoding:
    valid: list[SearchEncoding] = []
    for candidate in candidates:
        try:
            if decoder(candidate) == source:
                valid.append(candidate)
        except (ArithmeticError, EOFError, lzma.LZMAError, ValueError, zlib.error):
            continue
    if not valid:
        raise RuntimeError("no lossless encoding candidate was produced")
    return min(valid, key=lambda candidate: (candidate.encoded_bits, int(candidate.kind)))


def compress_bytes(data: bytes) -> SearchEncoding:
    """Search exact representations of arbitrary bytes and return the smallest.

    Raw storage is always a candidate, so incompressible data never expands by
    more than the format's six-to-ten-byte framing overhead.
    """
    source = bytes(data)
    candidates = [SearchEncoding(EncodingKind.RAW_BYTES, len(source), source)]
    period = _smallest_period(source)
    if period is not None:
        candidates.append(
            SearchEncoding(
                EncodingKind.PERIODIC_BYTES,
                len(source),
                _encode_varint(len(period)) + period,
            )
        )
    candidates.extend(
        (
            SearchEncoding(
                EncodingKind.ZLIB_BYTES,
                len(source),
                zlib.compress(source, level=9),
            ),
            SearchEncoding(
                EncodingKind.LZMA_BYTES,
                len(source),
                lzma.compress(source, format=lzma.FORMAT_XZ, preset=6),
            ),
        )
    )
    best = _verified_best(source, candidates, decode_bytes)
    raw = candidates[0]
    return best if best.kind != EncodingKind.RAW_BYTES and best.encoded_bits < len(source) * 8 else raw


def decode_bytes(encoding: SearchEncoding | bytes) -> bytes:
    """Decode an arbitrary-byte search encoding exactly."""
    candidate = parse_encoding(encoding) if isinstance(encoding, bytes) else encoding
    if candidate.kind == EncodingKind.RAW_BYTES:
        result = candidate.payload
    elif candidate.kind == EncodingKind.PERIODIC_BYTES:
        period_length, offset = _decode_varint(candidate.payload)
        period = candidate.payload[offset:]
        if period_length < 1 or len(period) != period_length:
            raise ValueError("invalid periodic byte payload")
        result = (period * math.ceil(candidate.original_length / period_length))[
            : candidate.original_length
        ]
    elif candidate.kind == EncodingKind.ZLIB_BYTES:
        result = zlib.decompress(candidate.payload)
    elif candidate.kind == EncodingKind.LZMA_BYTES:
        result = lzma.decompress(candidate.payload, format=lzma.FORMAT_XZ)
    else:
        raise ValueError(f"{candidate.kind.name} is not a byte encoding")
    if len(result) != candidate.original_length:
        raise ValueError("decoded byte length does not match header")
    return result


def _validate_digits(digits: str) -> None:
    if not digits or any(character < "0" or character > "9" for character in digits):
        raise ValueError("digits must be a non-empty ASCII decimal string")


def _decimal_to_integer(digits: str) -> int:
    """Convert arbitrary-length ASCII decimal without Python's global digit limit."""
    value = 0
    for offset in range(0, len(digits), 9):
        chunk = digits[offset : offset + 9]
        value = value * (10 ** len(chunk)) + int(chunk)
    return value


def _integer_to_decimal(value: int) -> str:
    """Convert a nonnegative integer to decimal in bounded base-1e9 chunks."""
    if value < 0:
        raise ValueError("cannot render a negative unsigned decimal value")
    if value == 0:
        return "0"
    chunks: list[int] = []
    while value:
        value, remainder = divmod(value, 1_000_000_000)
        chunks.append(remainder)
    return str(chunks[-1]) + "".join(f"{chunk:09d}" for chunk in reversed(chunks[:-1]))


def decimal_information_bits(digit_count: int) -> int:
    """Ceiling of the radix-10 information baseline without floating point."""
    if digit_count < 1:
        raise ValueError("digit_count must be positive")
    # 10**n has bit_length floor(n*log2(10))+1; ceil(log2(10**n)) is
    # one less only when 10**n is an exact power of two (never true for n>0).
    return (10**digit_count).bit_length()


def _raw_decimal_candidate(digits: str) -> SearchEncoding:
    return SearchEncoding(
        EncodingKind.RAW_DECIMAL,
        len(digits),
        _unsigned_bytes(_decimal_to_integer(digits)),
    )


def _rational_candidate(digits: str) -> SearchEncoding | None:
    """Find a compact continued-fraction convergent inside the decimal interval."""
    digit_count = len(digits)
    integer_digits = _decimal_to_integer(digits)
    scale = 10**digit_count
    midpoint = Fraction(2 * integer_digits + 1, 2 * scale)
    numerator, denominator = midpoint.numerator, midpoint.denominator
    previous_numerator, current_numerator = 0, 1
    previous_denominator, current_denominator = 1, 0
    while denominator:
        quotient, remainder = divmod(numerator, denominator)
        next_numerator = quotient * current_numerator + previous_numerator
        next_denominator = quotient * current_denominator + previous_denominator
        previous_numerator, current_numerator = current_numerator, next_numerator
        previous_denominator, current_denominator = current_denominator, next_denominator
        if current_denominator and (
            integer_digits * current_denominator
            <= current_numerator * scale
            < (integer_digits + 1) * current_denominator
        ):
            return SearchEncoding(
                EncodingKind.RATIONAL_DECIMAL,
                digit_count,
                _pack_integer(current_numerator) + _pack_integer(current_denominator),
            )
        numerator, denominator = denominator, remainder
    return None


def _modular_candidate(digits: str) -> SearchEncoding | None:
    if len(digits) < 4:
        return None
    values = [ord(character) - 48 for character in digits]
    for multiplier in range(10):
        increment = (values[1] - multiplier * values[0]) % 10
        if all(
            values[index + 1] == (multiplier * values[index] + increment) % 10
            for index in range(len(values) - 1)
        ):
            return SearchEncoding(
                EncodingKind.MODULAR_DECIMAL,
                len(digits),
                bytes((multiplier, increment, values[0])),
            )
    return None


def _power_offset_candidate(digits: str, maximum_offset: int = 65535) -> SearchEncoding | None:
    target = _decimal_to_integer(digits)
    if target < 2:
        return None
    best: SearchEncoding | None = None
    for base in range(2, 65):
        approximate_exponent = max(2, round(target.bit_length() / math.log2(base)))
        for exponent in range(max(2, approximate_exponent - 2), approximate_exponent + 3):
            offset = target - base**exponent
            if abs(offset) > maximum_offset:
                continue
            zigzag_offset = 2 * offset if offset >= 0 else -2 * offset - 1
            candidate = SearchEncoding(
                EncodingKind.POWER_OFFSET_DECIMAL,
                len(digits),
                _encode_varint(base)
                + _encode_varint(exponent)
                + _encode_varint(zigzag_offset),
            )
            if best is None or candidate.encoded_bits < best.encoded_bits:
                best = candidate
    return best


def compress_8000_digits(digits: str) -> SearchEncoding:
    """Search for an exact representation of one 8,000-decimal-digit block.

    The returned candidate is the smallest verified serialized representation.
    It may be raw radix-packed data when no discovered program is smaller.
    """
    _validate_digits(digits)
    if len(digits) != 8000:
        raise ValueError(f"expected exactly 8000 digits, got {len(digits)}")

    candidates = [_raw_decimal_candidate(digits)]
    digit_bytes = digits.encode("ascii")
    period = _smallest_period(digit_bytes)
    if period is not None:
        candidates.append(
            SearchEncoding(
                EncodingKind.PERIODIC_DECIMAL,
                len(digits),
                _encode_varint(len(period)) + period,
            )
        )
    for optional_candidate in (
        _rational_candidate(digits),
        _modular_candidate(digits),
        _power_offset_candidate(digits),
    ):
        if optional_candidate is not None:
            candidates.append(optional_candidate)

    packed_digits = _unsigned_bytes(_decimal_to_integer(digits))
    candidates.append(
        SearchEncoding(
            EncodingKind.LZMA_BYTES,
            len(digits),
            lzma.compress(packed_digits, format=lzma.FORMAT_XZ, preset=6),
        )
    )
    best = _verified_best(digits, candidates, decode_digits)
    raw = candidates[0]
    baseline_bits = decimal_information_bits(len(digits))
    return best if best.kind != EncodingKind.RAW_DECIMAL and best.encoded_bits < baseline_bits else raw


def decode_digits(encoding: SearchEncoding | bytes) -> str:
    """Decode a decimal search program using integer-only arithmetic."""
    candidate = parse_encoding(encoding) if isinstance(encoding, bytes) else encoding
    if candidate.kind == EncodingKind.RAW_DECIMAL:
        value = int.from_bytes(candidate.payload, "big")
    elif candidate.kind == EncodingKind.PERIODIC_DECIMAL:
        period_length, offset = _decode_varint(candidate.payload)
        period = candidate.payload[offset:]
        if period_length < 1 or len(period) != period_length:
            raise ValueError("invalid periodic decimal payload")
        result = (period * math.ceil(candidate.original_length / period_length))[
            : candidate.original_length
        ].decode("ascii")
        _validate_digits(result)
        return result
    elif candidate.kind == EncodingKind.RATIONAL_DECIMAL:
        numerator, offset = _unpack_integer(candidate.payload, 0)
        denominator, offset = _unpack_integer(candidate.payload, offset)
        if offset != len(candidate.payload) or denominator == 0:
            raise ValueError("invalid rational decimal payload")
        value = numerator * 10**candidate.original_length // denominator
    elif candidate.kind == EncodingKind.MODULAR_DECIMAL:
        if len(candidate.payload) != 3:
            raise ValueError("invalid modular decimal payload")
        multiplier, increment, value = candidate.payload
        if multiplier > 9 or increment > 9 or value > 9:
            raise ValueError("modular decimal parameters must be digits")
        output = []
        for _ in range(candidate.original_length):
            output.append(chr(48 + value))
            value = (multiplier * value + increment) % 10
        return "".join(output)
    elif candidate.kind == EncodingKind.POWER_OFFSET_DECIMAL:
        base, offset = _decode_varint(candidate.payload)
        exponent, offset = _decode_varint(candidate.payload, offset)
        zigzag_offset, offset = _decode_varint(candidate.payload, offset)
        if offset != len(candidate.payload):
            raise ValueError("trailing power-expression payload")
        signed_offset = (
            zigzag_offset // 2 if zigzag_offset % 2 == 0 else -(zigzag_offset // 2) - 1
        )
        value = base**exponent + signed_offset
    elif candidate.kind == EncodingKind.LZMA_BYTES:
        packed = lzma.decompress(candidate.payload, format=lzma.FORMAT_XZ)
        value = int.from_bytes(packed, "big")
    else:
        raise ValueError(f"{candidate.kind.name} is not a decimal encoding")

    result = _integer_to_decimal(value).zfill(candidate.original_length)
    if len(result) != candidate.original_length:
        raise ValueError("decoded value exceeds declared decimal length")
    return result
