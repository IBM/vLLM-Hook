"""Shared-memory helpers

  setup_shm()    — allocate the block + set env vars (called from HookLLM.__init__)
  teardown_shm() — close + unlink the block (called from HookLLM.__del__)
  load_from_shm()— poll ready flag, read tensors, return hs_cache dict
                   (called from HiddenStatesAnalyzer.analyze)
"""

from __future__ import annotations

import os
import struct
import time
import json
from typing import Dict, List, Optional, Any


def setup_shm(config_file: str, worker_name: str = None) -> Optional[Any]:
    """Allocate a SharedMemory block sized for (num_layers, max_batch, hidden_size).

    Reads hidden_size and target layers from config_file, allocates the block,
    sets all VLLM_HOOK_SHM_* env vars so the worker subprocess (forked after
    this call) can attach by name.

    Returns the SharedMemory object (caller must keep it alive), or None if
    the config is missing required fields or an unsupported worker/mode is
    detected (in which case VLLM_HOOK_USE_SHM is also cleared from the environment).
    """
    import warnings
    from multiprocessing.shared_memory import SharedMemory

    if worker_name != "probe_hidden_states":
        warnings.warn(
            f"VLLM_HOOK_USE_SHM=1 is only supported for 'probe_hidden_states', "
            f"got '{worker_name}' — SHM disabled.",
            UserWarning,
        )
        os.environ["VLLM_HOOK_USE_SHM"] = "0"
        return None

    hs_mode = os.environ.get("VLLM_HOOK_HS_MODE", "last_token")
    if hs_mode != "last_token":
        warnings.warn(
            f"VLLM_HOOK_USE_SHM=1 is only supported for 'last_token' mode, "
            f"got '{hs_mode}' — SHM disabled.",
            UserWarning,
        )
        os.environ["VLLM_HOOK_USE_SHM"] = "0"
        return None

    hidden_size = 0
    target_layers: list = []
    if config_file and os.path.exists(config_file):
        with open(config_file) as f:
            cfg = json.load(f)
        hs_cfg = cfg.get("hidden_states", {})
        target_layers = hs_cfg.get("layers", [])
        hidden_size = cfg.get("hidden_size", 0)

    if hidden_size <= 0 or not target_layers:
        print("VLLM_HOOK_USE_SHM=1 but could not read hidden_size / layers "
              "from config — SHM disabled.")
        os.environ.pop("VLLM_HOOK_USE_SHM", None)
        return None

    num_layers = len(target_layers)
    max_batch = int(os.environ.get("VLLM_HOOK_MAX_BATCH", "64"))
    # Header: 2×uint32 (num_layers, batch_size) = 8 bytes
    # Data:   float16 × num_layers × max_batch × hidden_size
    total_bytes = 8 + num_layers * max_batch * hidden_size * 2
    shm_name = f"vllm_hook_{os.getpid()}"

    shm = SharedMemory(create=True, size=total_bytes, name=shm_name)
    ready_flag = f"/dev/shm/vllm_hook_ready_{os.getpid()}"

    os.environ["VLLM_HOOK_SHM_NAME"] = shm_name
    os.environ["VLLM_HOOK_SHM_HIDDEN_SIZE"] = str(hidden_size)
    os.environ["VLLM_HOOK_SHM_NUM_LAYERS"] = str(num_layers)
    os.environ["VLLM_HOOK_SHM_MAX_BATCH"] = str(max_batch)
    os.environ["VLLM_HOOK_SHM_LAYER_ORDER"] = ";".join(map(str, sorted(target_layers)))
    os.environ["VLLM_HOOK_SHM_READY_FLAG"] = ready_flag

    return shm


def teardown_shm(shm: Optional[Any]) -> None:
    """Close and unlink a SharedMemory block returned by setup_shm().

    Safe to call with None (no-op).
    """
    if shm is None:
        return
    try:
        shm.close()
        shm.unlink()
    except Exception:
        pass


def load_from_shm(hook_dir: str, run_id_file: str) -> Dict:
    """Read tensors directly from the shared memory block.

    Polls for the ready flag written by the worker, then reads the buffer
    using the layout: [0:4] uint32 num_layers, [4:8] uint32 batch_size,
    [8:] float16 data row-major (layer_slot, batch_item, hidden_dim).

    Optionally persists the artifact to disk when VLLM_HOOK_SHM_PERSIST=1.
    """
    import torch
    from multiprocessing.shared_memory import SharedMemory

    ready_flag = os.environ["VLLM_HOOK_SHM_READY_FLAG"]
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

    shm = SharedMemory(create=False, name=os.environ["VLLM_HOOK_SHM_NAME"])
    hidden_size = int(os.environ["VLLM_HOOK_SHM_HIDDEN_SIZE"])
    layer_order = [
        int(x) for x in os.environ["VLLM_HOOK_SHM_LAYER_ORDER"].split(";") if x
    ]

    num_layers, batch_size = struct.unpack("II", bytes(shm.buf[0:8]))

    hs_cache: Dict = {}
    data_offset = 8
    for slot, lnum in enumerate(layer_order[:num_layers]):
        nbytes = batch_size * hidden_size * 2  # float16
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

    if os.environ.get("VLLM_HOOK_SHM_PERSIST", "0") == "1":
        from vllm_hook_plugins.run_utils import latest_run_id
        run_id = latest_run_id(run_id_file)
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

    return hs_cache
