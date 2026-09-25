"""Shared-memory buffers for moving captured tensors out of workers."""

from __future__ import annotations

import os
import struct
import time
import json
from typing import Dict, List, Optional, Any


def setup_shm(config_file: str, worker_name: str = None) -> Optional[Any]:
    """Allocate a SharedMemory block sized for (num_layers, max_batch, hidden_size)."""
    import warnings
    from multiprocessing.shared_memory import SharedMemory

    if worker_name != "capture_hs":
        warnings.warn(
            f"MIA_USE_SHM=1 is only supported for 'capture_hs', "
            f"got '{worker_name}' — SHM disabled.",
            UserWarning,
        )
        os.environ["MIA_USE_SHM"] = "0"
        return None

    hidden_size = 0
    target_layers: list = []
    hs_mode = "last_token"
    if config_file and os.path.exists(config_file):
        with open(config_file) as f:
            cfg = json.load(f)
        hs_cfg = cfg.get("hidden_states", {})
        target_layers = hs_cfg.get("layers", [])
        hidden_size = cfg.get("hidden_size", 0)
        hs_mode = hs_cfg.get("mode", "last_token")

    if hs_mode != "last_token":
        warnings.warn(
            f"MIA_USE_SHM=1 is only supported for 'last_token' mode, "
            f"got '{hs_mode}' — SHM disabled.",
            UserWarning,
        )
        os.environ["MIA_USE_SHM"] = "0"
        return None

    if hidden_size <= 0 or not target_layers:
        print("MIA_USE_SHM=1 but could not read hidden_size / layers "
              "from config — SHM disabled.")
        os.environ.pop("MIA_USE_SHM", None)
        return None

    num_layers = len(target_layers)
    max_batch = int(os.environ.get("MIA_MAX_BATCH", "64"))
    total_bytes = 8 + num_layers * max_batch * hidden_size * 2
    shm_name = f"mia_{os.getpid()}"

    shm = SharedMemory(create=True, size=total_bytes, name=shm_name)
    ready_flag = f"/dev/shm/mia_ready_{os.getpid()}"

    os.environ["MIA_SHM_NAME"] = shm_name
    os.environ["MIA_SHM_HIDDEN_SIZE"] = str(hidden_size)
    os.environ["MIA_SHM_NUM_LAYERS"] = str(num_layers)
    os.environ["MIA_SHM_MAX_BATCH"] = str(max_batch)
    os.environ["MIA_SHM_LAYER_ORDER"] = ";".join(map(str, sorted(target_layers)))
    os.environ["MIA_SHM_READY_FLAG"] = ready_flag

    return shm


def teardown_shm(shm: Optional[Any]) -> None:
    """Close and unlink a SharedMemory block returned by setup_shm()."""
    if shm is None:
        return
    try:
        shm.close()
        shm.unlink()
    except Exception:
        pass


def load_from_shm(hook_dir: str, run_id: Optional[str] = None) -> Dict:
    """Read tensors directly from the shared memory block."""
    import torch
    from multiprocessing.shared_memory import SharedMemory

    ready_flag = os.environ["MIA_SHM_READY_FLAG"]
    deadline = time.monotonic() + 10.0
    while not os.path.exists(ready_flag):
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"SHM ready flag not set within 10 s: {ready_flag}"
            )
        time.sleep(0.005)
    with open(ready_flag) as _f:
        _flag_content = _f.read().strip()
    peak_gpu_mb = float(_flag_content) if _flag_content else 0.0
    os.unlink(ready_flag)

    shm = SharedMemory(create=False, name=os.environ["MIA_SHM_NAME"])
    hidden_size = int(os.environ["MIA_SHM_HIDDEN_SIZE"])
    layer_order = [
        int(x) for x in os.environ["MIA_SHM_LAYER_ORDER"].split(";") if x
    ]

    num_layers, batch_size = struct.unpack("II", bytes(shm.buf[0:8]))

    hs_cache: Dict = {}
    data_offset = 8
    for slot, lnum in enumerate(layer_order[:num_layers]):
        nbytes = batch_size * hidden_size * 2
        start = data_offset + slot * batch_size * hidden_size * 2
        raw = bytes(shm.buf[start : start + nbytes])
        t = torch.frombuffer(bytearray(raw), dtype=torch.float16).view(
            batch_size, hidden_size
        )
        layer_name = f"model.layers.{lnum}"
        hs_cache[layer_name] = {
            "hidden_states": [t[i].clone() for i in range(batch_size)],
            "layer_num": lnum,
        }

    shm.close()

    if os.environ.get("MIA_SHM_PERSIST", "0") == "1":
        if not run_id:
            raise ValueError("MIA_SHM_PERSIST=1 requires a run_id passed to load_from_shm.")
        run_dir = os.path.join(hook_dir, run_id, "tp_rank_0")
        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, "hidden_states.pt")
        tmp_path = out_path + ".tmp"
        cpu_cache = {
            "config": {"hidden_size": hidden_size, "num_layers": len(layer_order)},
            "hs_cache": hs_cache,
            "peak_gpu_mb": peak_gpu_mb,
        }
        with open(tmp_path, "wb") as f:
            torch.save(cpu_cache, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, out_path)

    return hs_cache, peak_gpu_mb

