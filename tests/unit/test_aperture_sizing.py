"""CPU-only checks of the aperture fit rule: the aperture comes out of gpu_memory_utilization."""
import pytest

from mia.core.aperture.aperture_sizing import (
    DEFAULT_APERTURE_GPU_BYTES, resolve_aperture_bytes_auto, resolve_aperture_bytes_fixed)
from mia.errors import MiaSizingError

GiB, MiB = 1 << 30, 1 << 20
CARD_6GB = int(5.64 * GiB)      # RTX 4050 Laptop as torch reports it
WEIGHTS_1P5B = int(2.98 * GiB)  # Qwen2-1.5B fp16


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("MIA_APERTURE_GPU_BYTES", raising=False)


def test_small_aperture_fits_inside_share_on_small_card():
    for nbytes in (128 * MiB, 512 * MiB, 1 * GiB):
        assert resolve_aperture_bytes_fixed(CARD_6GB, nbytes, 0.85, WEIGHTS_1P5B) == nbytes


def test_aperture_larger_than_share_left_after_weights_is_refused():
    with pytest.raises(MiaSizingError, match="comes out of vLLM's gpu_memory_utilization"):
        resolve_aperture_bytes_fixed(CARD_6GB, 2 * GiB, 0.85, WEIGHTS_1P5B)


def test_raising_gpu_memory_utilization_makes_room():
    with pytest.raises(MiaSizingError):
        resolve_aperture_bytes_fixed(CARD_6GB, int(1.5 * GiB), 0.75, WEIGHTS_1P5B)
    assert resolve_aperture_bytes_fixed(CARD_6GB, int(1.5 * GiB), 0.85, WEIGHTS_1P5B)


def test_high_gpu_memory_utilization_no_longer_refused_on_large_card():
    # was refused when the aperture was assumed to live outside the share (2.4 GiB < 4 GiB)
    card_80 = 80 * GiB
    assert resolve_aperture_bytes_fixed(card_80, DEFAULT_APERTURE_GPU_BYTES, 0.97) == 4 * GiB


def test_explicit_env_wins_and_is_checked(monkeypatch):
    monkeypatch.setenv("MIA_APERTURE_GPU_BYTES", str(256 * MiB))
    assert resolve_aperture_bytes_auto(CARD_6GB, 0.85, weights_bytes=WEIGHTS_1P5B) == 256 * MiB
    monkeypatch.setenv("MIA_APERTURE_GPU_BYTES", str(3 * GiB))
    with pytest.raises(MiaSizingError):
        resolve_aperture_bytes_auto(CARD_6GB, 0.85, weights_bytes=WEIGHTS_1P5B)


def test_default_grows_to_one_step_within_share():
    rows, row_bytes = 65536, 28 * 3584 * 2  # 7B-ish, 28 layers: 12.25 GiB per step
    got = resolve_aperture_bytes_auto(80 * GiB, 0.9, rows_needed=rows, row_bytes=row_bytes,
                                      weights_bytes=15 * GiB)
    assert got == rows * row_bytes
    with pytest.raises(MiaSizingError, match="one max-token step"):
        resolve_aperture_bytes_auto(24 * GiB, 0.9, rows_needed=rows, row_bytes=row_bytes,
                                    weights_bytes=15 * GiB)
