"""Contract tests for the exact program-search compressor."""

import random

import pytest

from src.search_compressor import (
    EncodingKind,
    compress_8000_digits,
    compress_bytes,
    decimal_information_bits,
    decode_bytes,
    decode_digits,
    parse_encoding,
)


def test_periodic_8000_digit_program_is_tiny_and_exact():
    digits = ("142857" * 1334)[:8000]
    encoding = compress_8000_digits(digits)

    # 142857... is both periodic and the decimal expansion of 1/7; the
    # serializer correctly chooses the smaller of those exact programs.
    assert encoding.kind in {
        EncodingKind.PERIODIC_DECIMAL,
        EncodingKind.RATIONAL_DECIMAL,
    }
    assert decode_digits(encoding) == digits
    assert decode_digits(encoding.serialize()) == digits
    assert encoding.encoded_bits < decimal_information_bits(8000)


def test_modular_decimal_generator_is_found_and_exact():
    values = []
    value = 7
    for _ in range(8000):
        values.append(str(value))
        value = (7 * value + 3) % 10
    digits = "".join(values)

    encoding = compress_8000_digits(digits)

    # This sequence is also eventually periodic, and the encoder is free to
    # choose whichever verified serialized program is smaller.
    assert encoding.kind in {
        EncodingKind.PERIODIC_DECIMAL,
        EncodingKind.MODULAR_DECIMAL,
    }
    assert decode_digits(encoding.serialize()) == digits


def test_random_digits_fall_back_losslessly():
    random_source = random.Random(23)
    digits = "".join(str(random_source.randrange(10)) for _ in range(8000))

    encoding = compress_8000_digits(digits)

    assert decode_digits(encoding.serialize()) == digits
    # Raw radix packing should stay close to the 26,576-bit information bound.
    assert encoding.encoded_bits <= decimal_information_bits(8000) + 80


def test_leading_zero_digits_roundtrip():
    digits = "0" * 7990 + "1234567890"
    encoding = compress_8000_digits(digits)
    assert decode_digits(encoding.serialize()) == digits


def test_power_expression_roundtrip():
    digits = str(3**503 + 17).zfill(8000)
    encoding = compress_8000_digits(digits)
    assert encoding.kind == EncodingKind.POWER_OFFSET_DECIMAL
    assert decode_digits(encoding) == digits


def test_binary_periodic_and_incompressible_roundtrips():
    periodic = (b"\x00\x7f\xff" * 4096)[:8000]
    periodic_encoding = compress_bytes(periodic)
    assert periodic_encoding.kind == EncodingKind.PERIODIC_BYTES
    assert decode_bytes(periodic_encoding.serialize()) == periodic
    assert periodic_encoding.encoded_bytes < len(periodic)

    random_source = random.Random(91)
    random_bytes = random_source.randbytes(8000)
    random_encoding = compress_bytes(random_bytes)
    assert random_encoding.kind == EncodingKind.RAW_BYTES
    assert decode_bytes(random_encoding.serialize()) == random_bytes


def test_parser_rejects_bad_framing():
    with pytest.raises(ValueError, match="magic"):
        parse_encoding(b"bad")
    with pytest.raises(ValueError, match="unknown"):
        parse_encoding(b"LSC1\xff\x01")


def test_requires_exactly_8000_decimal_digits():
    with pytest.raises(ValueError, match="exactly 8000"):
        compress_8000_digits("1" * 7999)
    with pytest.raises(ValueError, match="ASCII decimal"):
        compress_8000_digits("x" * 8000)
