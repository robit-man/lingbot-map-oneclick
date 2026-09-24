from __future__ import annotations

import pytest

from webapp.gpu_selection import GpuSelectionError, select_gpu


DISCOVERY = {
    "selected_gpu_ids": ["GPU-0", "GPU-1", "GPU-2"],
    "gpus": [
        {"uuid": "GPU-0", "free_mib": 12000},
        {"uuid": "GPU-1", "free_mib": 8000},
        {"uuid": "GPU-2", "free_mib": 16000},
        {"uuid": "GPU-3", "free_mib": 32000},
    ],
}


def test_gpu_index_resolves_to_exact_broker_uuid():
    assert select_gpu(DISCOVERY, requested_index="1") == "GPU-1"


def test_auto_selects_most_free_broker_eligible_gpu():
    assert select_gpu(DISCOVERY, requested_index="auto") == "GPU-2"


def test_uuid_constraint_remains_supported():
    assert select_gpu(DISCOVERY, requested_uuid="GPU-0") == "GPU-0"


@pytest.mark.parametrize("requested_index", ["3", "9", "gpu1", "-1"])
def test_invalid_or_ineligible_gpu_index_fails_closed(requested_index):
    with pytest.raises(GpuSelectionError):
        select_gpu(DISCOVERY, requested_index=requested_index)


def test_ambiguous_uuid_and_index_constraints_fail_closed():
    with pytest.raises(GpuSelectionError, match="only one"):
        select_gpu(
            DISCOVERY,
            requested_uuid="GPU-0",
            requested_index="1",
        )
