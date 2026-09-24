from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class GpuSelectionError(ValueError):
    """Raised when a configured GPU constraint cannot become a broker scope."""


def _normalize_index(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "auto":
        return None
    if not text.isdecimal():
        raise GpuSelectionError(
            f"configured GPU index must be a non-negative integer or auto: {text}"
        )
    return int(text)


def select_gpu(
    discovery: Mapping[str, Any],
    *,
    requested_uuid: str | None = None,
    requested_index: str | int | None = None,
) -> str:
    """Resolve an operator selector to one exact broker-eligible GPU UUID."""

    uuid_constraint = str(requested_uuid or "").strip()
    index_constraint = _normalize_index(requested_index)
    if uuid_constraint and index_constraint is not None:
        raise GpuSelectionError(
            "configure only one of LINGBOT_GPU_UUID or LINGBOT_GPU_INDEX"
        )

    gpus = [gpu for gpu in discovery.get("gpus", []) if isinstance(gpu, Mapping)]
    selected_ids = {
        str(uuid)
        for uuid in discovery.get("selected_gpu_ids", [])
        if str(uuid).strip()
    }

    if uuid_constraint:
        if uuid_constraint not in selected_ids:
            raise GpuSelectionError(
                f"configured GPU is not broker-selected: {uuid_constraint}"
            )
        return uuid_constraint

    if index_constraint is not None:
        if index_constraint >= len(gpus):
            raise GpuSelectionError(
                f"configured GPU index is unavailable: {index_constraint}"
            )
        resolved = str(gpus[index_constraint].get("uuid") or "").strip()
        if not resolved or resolved not in selected_ids:
            raise GpuSelectionError(
                f"configured GPU index {index_constraint} is not broker-selected"
            )
        return resolved

    eligible = [gpu for gpu in gpus if str(gpu.get("uuid") or "") in selected_ids]
    if not eligible:
        raise GpuSelectionError("docker gpu discover returned no eligible scoped GPU")
    return str(max(eligible, key=lambda gpu: int(gpu.get("free_mib") or 0))["uuid"])


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print(
            "usage: gpu_selection.py DISCOVERY_JSON GPU_UUID GPU_INDEX",
            file=sys.stderr,
        )
        return 2
    discovery_path, requested_uuid, requested_index = args
    try:
        discovery = json.loads(Path(discovery_path).read_text(encoding="utf-8"))
        print(
            select_gpu(
                discovery,
                requested_uuid=requested_uuid,
                requested_index=requested_index,
            )
        )
    except (OSError, json.JSONDecodeError, GpuSelectionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
