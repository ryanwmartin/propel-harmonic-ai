"""Contract tests for the adaptive arithmetic entropy coder."""

import random

import pytest

from src.entropy_coder import decode_bytes, encode_bytes, measure_encoding


@pytest.mark.parametrize(
    "source",
    [
        b"",
        b"A",
        b"ABABAAC",
        bytes(range(256)),
        b"\x00" * 8192,
        (b"phasor-inference" * 1024),
    ],
)
def test_entropy_coder_roundtrips_exactly(source):
    encoded = encode_bytes(source)
    assert decode_bytes(encoded) == source


def test_entropy_coder_roundtrips_deterministic_random_bytes():
    source = random.Random(137).randbytes(65_537)
    encoded = encode_bytes(source)

    assert decode_bytes(encoded) == source
    metrics = measure_encoding(source, encoded)
    assert metrics.source_bytes == len(source)
    assert metrics.encoded_bytes == len(encoded)
    # Adaptive coding cannot compress uniform random data, but overhead remains
    # bounded to model adaptation and stream framing rather than hidden state.
    assert metrics.encoded_to_source_ratio < 1.02


def test_low_entropy_bytes_compress_and_roundtrip():
    source = b"\x00" * 100_000
    encoded = encode_bytes(source)
    metrics = measure_encoding(source, encoded)

    assert decode_bytes(encoded) == source
    assert metrics.encoded_to_source_ratio < 0.02


def test_fp16_storage_metric():
    source = b"\x00\x3c" * 100
    metrics = measure_encoding(source, encode_bytes(source))
    assert metrics.bits_per_fp16_weight == metrics.encoded_bytes * 8 / 100


def test_decoder_rejects_bad_framing_and_corruption():
    with pytest.raises(ValueError, match="truncated"):
        decode_bytes(b"")

    encoded = bytearray(encode_bytes(b"ABABAAC" * 100))
    encoded[-1] ^= 0x80
    with pytest.raises(ValueError, match="checksum"):
        decode_bytes(bytes(encoded))
