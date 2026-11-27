"""Cross-platform host hardware profiling, scaling overrides, and cross-process slot gate."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import ctypes
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, AsyncIterator, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

DEFAULT_KV_CACHE_TYPE = "q8_0"
DEFAULT_FLASH_ATTENTION = 1
DEFAULT_FALLBACK_RAM_GB = 16.0

_TRUTHY_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class HostHardware:
    """Detected hardware attributes for the host running Ollama."""

    os_name: str
    arch: str
    cpu_brand: str
    logical_cores: int
    perf_cores: int | None
    total_ram_gb: float
    gpu_vram_gb: float | None
    is_unified_memory: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize hardware metadata into a JSON-compatible dictionary."""
        return asdict(self)


def _run_short_command(cmd: list[str]) -> str:
    """Run a short read-only command and return stripped stdout, or empty string on error."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _parse_positive_int(raw: str) -> int | None:
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def _detect_macos_hardware(arch: str, fallback_cores: int) -> tuple[str, int, int | None, float, bool]:
    cpu_brand = _run_short_command(["sysctl", "-n", "machdep.cpu.brand_string"]) or (
        platform.processor() or arch
    )
    ncpu = _parse_positive_int(_run_short_command(["sysctl", "-n", "hw.ncpu"]))
    perf_cores = _parse_positive_int(
        _run_short_command(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
    )
    mem_bytes = _parse_positive_int(_run_short_command(["sysctl", "-n", "hw.memsize"]))
    total_ram_gb = (
        round(mem_bytes / (1024**3), 2) if mem_bytes is not None else DEFAULT_FALLBACK_RAM_GB
    )
    is_unified = arch.lower() in {"arm64", "aarch64"}
    return cpu_brand, (ncpu or fallback_cores), perf_cores, total_ram_gb, is_unified


def _detect_linux_hardware(arch: str, fallback_cores: int) -> tuple[str, int, float, float | None]:
    cpu_brand = platform.processor() or arch
    total_ram_gb = DEFAULT_FALLBACK_RAM_GB
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        if kb > 0:
                            total_ram_gb = round(kb / (1024**2), 2)
                    break
    except OSError:
        pass

    gpu_vram_gb: float | None = None
    nvidia_out = _run_short_command(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
    )
    if nvidia_out:
        vram_mib_values: list[float] = []
        for line in nvidia_out.splitlines():
            try:
                mib = float(line.strip())
                if mib > 0:
                    vram_mib_values.append(mib)
            except ValueError:
                continue
        if vram_mib_values:
            gpu_vram_gb = round(max(vram_mib_values) / 1024.0, 2)

    return cpu_brand, fallback_cores, total_ram_gb, gpu_vram_gb


def _detect_windows_ram_gb() -> float:
    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        kernel32 = getattr(ctypes, "windll", None)
        if kernel32 is not None and kernel32.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            if status.ullTotalPhys > 0:
                return round(status.ullTotalPhys / (1024**3), 2)
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_FALLBACK_RAM_GB


def detect_host_hardware() -> HostHardware:
    """Detect host OS, CPU, RAM, and optional GPU VRAM safely without raising."""
    os_name = platform.system() or "Unknown"
    arch = platform.machine() or "unknown"
    fallback_cores = max(1, os.cpu_count() or 1)

    cpu_brand = platform.processor() or arch
    logical_cores = fallback_cores
    perf_cores: int | None = None
    total_ram_gb = DEFAULT_FALLBACK_RAM_GB
    gpu_vram_gb: float | None = None
    is_unified_memory = False

    try:
        if os_name == "Darwin":
            (
                cpu_brand,
                logical_cores,
                perf_cores,
                total_ram_gb,
                is_unified_memory,
            ) = _detect_macos_hardware(arch, fallback_cores)
        elif os_name == "Linux":
            cpu_brand, logical_cores, total_ram_gb, gpu_vram_gb = _detect_linux_hardware(
                arch, fallback_cores
            )
        elif os_name == "Windows":
            total_ram_gb = _detect_windows_ram_gb()
    except Exception:  # noqa: BLE001
        pass

    return HostHardware(
        os_name=os_name,
        arch=arch,
        cpu_brand=cpu_brand,
        logical_cores=max(1, logical_cores),
        perf_cores=perf_cores,
        total_ram_gb=max(1.0, float(total_ram_gb)),
        gpu_vram_gb=gpu_vram_gb,
        is_unified_memory=is_unified_memory,
    )


@dataclass(frozen=True)
class WorkerScalingProfile:
    """Computed concurrency, daemon tuning, model recommendations, and warnings."""

    hardware: HostHardware
    tier: str
    effective_memory_gb: float
    effective_cores: int
    max_parallel_workers: int
    recommended_ollama_num_parallel: int
    recommended_ollama_max_loaded_models: int
    recommended_context_length: int
    recommended_kv_cache_type: str
    recommended_flash_attention: int
    recommended_single_model: str
    recommended_parallel_model: str
    overrides_applied: dict[str, str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize scaling profile and hardware metadata to JSON-friendly dict."""
        data = asdict(self)
        data["hardware"] = self.hardware.to_dict()
        return data

    def format_summary(self) -> str:
        """Render a concise, human-readable scaling and warning summary."""
        lines = [
            f"Host Scaling Profile (tier={self.tier}):",
            (
                f"- Hardware: {self.hardware.cpu_brand} ({self.hardware.os_name}/{self.hardware.arch}), "
                f"effective_cores={self.effective_cores}, effective_memory={self.effective_memory_gb:.1f} GB"
            ),
            f"- Worker Concurrency: max_parallel_workers={self.max_parallel_workers}",
            (
                "- Recommended Ollama Daemon Env: "
                f"OLLAMA_NUM_PARALLEL={self.recommended_ollama_num_parallel}, "
                f"OLLAMA_MAX_LOADED_MODELS={self.recommended_ollama_max_loaded_models}, "
                f"OLLAMA_CONTEXT_LENGTH={self.recommended_context_length}, "
                f"OLLAMA_KV_CACHE_TYPE={self.recommended_kv_cache_type}, "
                f"OLLAMA_FLASH_ATTENTION={self.recommended_flash_attention}"
            ),
            (
                f"- Recommended Models: single={self.recommended_single_model} | "
                f"parallel_subagents={self.recommended_parallel_model}"
            ),
        ]
        if self.overrides_applied:
            formatted = ", ".join(f"{k}={v}" for k, v in sorted(self.overrides_applied.items()))
            lines.append(f"- Overrides Applied: {formatted}")
        if self.warnings:
            lines.append("- Warnings:")
            for warning in self.warnings:
                lines.append(f"  * {warning}")
        return "\n".join(lines)


def build_scaling_profile(
    hardware: HostHardware | None = None,
    env: Mapping[str, str] | None = None,
) -> WorkerScalingProfile:
    """Compute a WorkerScalingProfile from host hardware and optional OLLAMA_WORKER_* overrides."""
    hw = hardware or detect_host_hardware()
    environ = os.environ if env is None else env

    if not hw.is_unified_memory and hw.gpu_vram_gb is not None and hw.gpu_vram_gb > 0:
        effective_memory_gb = float(hw.gpu_vram_gb)
    else:
        effective_memory_gb = float(hw.total_ram_gb)

    effective_cores = (
        hw.perf_cores if (hw.perf_cores is not None and hw.perf_cores >= 1) else max(1, hw.logical_cores)
    )

    overrides_applied: dict[str, str] = {}
    warnings: list[str] = []

    raw_mem = (environ.get("OLLAMA_WORKER_MEMORY_GB") or "").strip()
    if raw_mem:
        try:
            parsed_mem = float(raw_mem)
            if parsed_mem > 0:
                effective_memory_gb = parsed_mem
                overrides_applied["OLLAMA_WORKER_MEMORY_GB"] = raw_mem
            else:
                warnings.append(
                    f"Ignored invalid OLLAMA_WORKER_MEMORY_GB={raw_mem!r} (must be > 0)."
                )
        except ValueError:
            warnings.append(
                f"Ignored malformed OLLAMA_WORKER_MEMORY_GB={raw_mem!r} (expected number)."
            )

    max_cores_override: int | None = None
    raw_cores = (environ.get("OLLAMA_WORKER_MAX_CORES") or "").strip()
    if raw_cores:
        try:
            parsed_cores = int(raw_cores)
            if parsed_cores >= 1:
                max_cores_override = parsed_cores
                effective_cores = parsed_cores
                overrides_applied["OLLAMA_WORKER_MAX_CORES"] = raw_cores
            else:
                warnings.append(
                    f"Ignored invalid OLLAMA_WORKER_MAX_CORES={raw_cores!r} (must be >= 1)."
                )
        except ValueError:
            warnings.append(
                f"Ignored malformed OLLAMA_WORKER_MAX_CORES={raw_cores!r} (expected integer)."
            )

    max_parallel_override: int | None = None
    raw_parallel = (environ.get("OLLAMA_WORKER_MAX_PARALLEL") or "").strip()
    if raw_parallel:
        try:
            parsed_parallel = int(raw_parallel)
            if parsed_parallel >= 1:
                max_parallel_override = parsed_parallel
                overrides_applied["OLLAMA_WORKER_MAX_PARALLEL"] = raw_parallel
            else:
                warnings.append(
                    f"Ignored invalid OLLAMA_WORKER_MAX_PARALLEL={raw_parallel!r} (must be >= 1)."
                )
        except ValueError:
            warnings.append(
                f"Ignored malformed OLLAMA_WORKER_MAX_PARALLEL={raw_parallel!r} (expected integer)."
            )

    ctx_override: int | None = None
    raw_ctx = (environ.get("OLLAMA_WORKER_CONTEXT_LENGTH") or "").strip()
    if raw_ctx:
        try:
            parsed_ctx = int(raw_ctx)
            if parsed_ctx >= 4096:
                ctx_override = parsed_ctx
                overrides_applied["OLLAMA_WORKER_CONTEXT_LENGTH"] = raw_ctx
            else:
                warnings.append(
                    f"Ignored invalid OLLAMA_WORKER_CONTEXT_LENGTH={raw_ctx!r} (must be >= 4096)."
                )
        except ValueError:
            warnings.append(
                f"Ignored malformed OLLAMA_WORKER_CONTEXT_LENGTH={raw_ctx!r} (expected integer)."
            )

    if effective_memory_gb <= 18.0:
        tier = "compact-<=16gb"
        default_parallel = 1
        max_loaded = 1
        ctx = 32768
        single_model = "gemma4-32k:latest"
        parallel_model = "qwen3:4b"
    elif effective_memory_gb < 44.0:
        tier = "mid-24gb-36gb"
        default_parallel = 2 if effective_cores >= 2 else 1
        max_loaded = 1 if effective_memory_gb < 30.0 else 2
        ctx = 32768
        single_model = "gemma4-32k:latest"
        parallel_model = "gemma4-32k:latest"
    else:
        tier = "high-48gb+"
        default_parallel = min(4, max(2, effective_cores // 2))
        max_loaded = 2
        ctx = 65536
        single_model = "qwen2.5-coder:14b"
        parallel_model = "gemma4-32k:latest"

    if max_cores_override is not None and max_parallel_override is None:
        default_parallel = min(default_parallel, max_cores_override)

    max_parallel_workers = (
        max_parallel_override if max_parallel_override is not None else default_parallel
    )

    if ctx_override is not None:
        ctx = ctx_override

    raw_model = (environ.get("OLLAMA_WORKER_MODEL") or "").strip()
    if raw_model:
        single_model = raw_model
        overrides_applied["OLLAMA_WORKER_MODEL"] = raw_model

    raw_parallel_model = (environ.get("OLLAMA_WORKER_PARALLEL_MODEL") or "").strip()
    if raw_parallel_model:
        parallel_model = raw_parallel_model
        overrides_applied["OLLAMA_WORKER_PARALLEL_MODEL"] = raw_parallel_model

    if single_model.lower().startswith("qwen2.5-coder") or parallel_model.lower().startswith(
        "qwen2.5-coder"
    ):
        warnings.append(
            "qwen2.5-coder frequently emits raw JSON text instead of structured OpenAI tool_calls "
            "under LocalOpenAIAgentConfig (causing no-op tool runs); prefer gemma4-32k:latest for "
            "reliable tool calling or qwen3:4b / gemma4:e4b for small parallel slots."
        )

    if effective_memory_gb <= 18.0 and max_parallel_workers >= 2:
        warnings.append(
            "OLLAMA_NUM_PARALLEL multiplies KV cache allocation across active slots; on <=16 GB "
            f"hosts use a <=5 GB native tool-calling model (e.g. {parallel_model}) for concurrent "
            "subagents and keep OLLAMA_MAX_LOADED_MODELS=1 to avoid CPU/swap spill and model thrashing."
        )

    if max_parallel_workers > effective_cores or (
        effective_memory_gb <= 18.0 and max_parallel_workers > 2
    ):
        warnings.append(
            f"Concurrency oversubscription warning: max_parallel_workers={max_parallel_workers} "
            f"exceeds safe budget for effective_cores={effective_cores}, "
            f"effective_memory_gb={effective_memory_gb:.1f}."
        )

    if max_parallel_workers >= 2:
        warnings.append(
            f"Ollama daemon defaults to OLLAMA_NUM_PARALLEL=1; set "
            f"OLLAMA_NUM_PARALLEL={max_parallel_workers} (e.g. via `./install_plugin.sh --tune-ollama`) "
            "and restart Ollama so concurrent subagents execute in parallel instead of queueing."
        )

    return WorkerScalingProfile(
        hardware=hw,
        tier=tier,
        effective_memory_gb=round(effective_memory_gb, 2),
        effective_cores=effective_cores,
        max_parallel_workers=max_parallel_workers,
        recommended_ollama_num_parallel=max_parallel_workers,
        recommended_ollama_max_loaded_models=max_loaded,
        recommended_context_length=ctx,
        recommended_kv_cache_type=DEFAULT_KV_CACHE_TYPE,
        recommended_flash_attention=DEFAULT_FLASH_ATTENTION,
        recommended_single_model=single_model,
        recommended_parallel_model=parallel_model,
        overrides_applied=overrides_applied,
        warnings=warnings,
    )


@dataclass(frozen=True)
class SlotLease:
    """Active concurrency slot lease returned by WorkerSlotGate."""

    slot_index: int
    active_slots: int
    workspace_collision: bool


class WorkerSlotGate:
    """Cross-process concurrency gate using fcntl.flock with asyncio.Semaphore fallback."""

    def __init__(
        self,
        max_slots: int,
        lock_dir: str | Path | None = None,
        disable_file_lock: bool = False,
    ) -> None:
        self.max_slots = max(1, int(max_slots))
        env_dir = (os.environ.get("OLLAMA_WORKER_LOCK_DIR") or "").strip()
        self.lock_dir = (
            Path(lock_dir)
            if lock_dir is not None
            else (Path(env_dir) if env_dir else Path(tempfile.gettempdir()) / "ollama-worker-locks")
        )
        disable_env = (os.environ.get("OLLAMA_WORKER_DISABLE_SLOT_LOCK") or "").strip().lower()
        self._use_file_lock = (
            fcntl is not None and not disable_file_lock and disable_env not in _TRUTHY_VALUES
        )
        self._semaphore = asyncio.Semaphore(self.max_slots)
        self._active_workspaces: dict[int, str] = {}

    @staticmethod
    def _normalize_workspace(workspace_path: str) -> str:
        cleaned = (workspace_path or "").strip()
        if not cleaned:
            return ""
        try:
            return os.path.realpath(cleaned)
        except OSError:
            return cleaned

    def _slot_lock_path(self, index: int) -> Path:
        return self.lock_dir / f"slot-{index}.lock"

    def _slot_ws_path(self, index: int) -> Path:
        return self.lock_dir / f"slot-{index}.ws"

    def _inspect_active_slots_file(self, own_index: int, norm_ws: str) -> tuple[int, bool]:
        active_slots = 1
        collision = False
        for idx in range(self.max_slots):
            if idx == own_index:
                continue
            lock_path = self._slot_lock_path(idx)
            if not lock_path.exists():
                continue
            fd: int | None = None
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[union-attr]
                # Acquired lock means slot idx is NOT held by another process; clean stale .ws.
                self._slot_ws_path(idx).unlink(missing_ok=True)
                fcntl.flock(fd, fcntl.LOCK_UN)  # type: ignore[union-attr]
            except BlockingIOError:
                active_slots += 1
                if norm_ws:
                    try:
                        other_ws = self._slot_ws_path(idx).read_text(encoding="utf-8").strip()
                        if other_ws == norm_ws:
                            collision = True
                    except OSError:
                        pass
            except OSError:
                pass
            finally:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        return active_slots, collision

    @asynccontextmanager
    async def acquire(self, workspace_path: str = "") -> AsyncIterator[SlotLease]:
        """Acquire one worker execution slot, yielding SlotLease with collision info."""
        norm_ws = self._normalize_workspace(workspace_path)
        if not self._use_file_lock:
            async with self._semaphore:
                slot_idx = next(
                    (i for i in range(self.max_slots) if i not in self._active_workspaces),
                    0,
                )
                collision = bool(
                    norm_ws
                    and any(
                        ws == norm_ws for i, ws in self._active_workspaces.items() if i != slot_idx
                    )
                )
                self._active_workspaces[slot_idx] = norm_ws
                try:
                    yield SlotLease(
                        slot_index=slot_idx,
                        active_slots=len(self._active_workspaces),
                        workspace_collision=collision,
                    )
                finally:
                    self._active_workspaces.pop(slot_idx, None)
            return

        self.lock_dir.mkdir(parents=True, exist_ok=True)
        acquired_fd: int | None = None
        acquired_idx = -1

        try:
            while acquired_fd is None:
                for idx in range(self.max_slots):
                    fd = os.open(self._slot_lock_path(idx), os.O_CREAT | os.O_RDWR, 0o600)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[union-attr]
                        acquired_fd = fd
                        acquired_idx = idx
                        break
                    except BlockingIOError:
                        os.close(fd)
                    except Exception:
                        os.close(fd)
                        raise
                if acquired_fd is None:
                    await asyncio.sleep(0.05)

            self._slot_ws_path(acquired_idx).write_text(norm_ws, encoding="utf-8")
            active_slots, collision = self._inspect_active_slots_file(acquired_idx, norm_ws)
            yield SlotLease(
                slot_index=acquired_idx,
                active_slots=active_slots,
                workspace_collision=collision,
            )
        finally:
            if acquired_idx >= 0:
                self._slot_ws_path(acquired_idx).unlink(missing_ok=True)
            if acquired_fd is not None:
                try:
                    fcntl.flock(acquired_fd, fcntl.LOCK_UN)  # type: ignore[union-attr]
                except OSError:
                    pass
                try:
                    os.close(acquired_fd)
                except OSError:
                    pass


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect host hardware and ollama-worker scaling profile."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the resolved scaling profile as JSON.",
    )
    args = parser.parse_args()
    profile = build_scaling_profile()
    if args.json:
        print(json.dumps(profile.to_dict(), indent=2))
    else:
        print(profile.format_summary())
    return 0


__all__ = [
    "DEFAULT_FALLBACK_RAM_GB",
    "DEFAULT_FLASH_ATTENTION",
    "DEFAULT_KV_CACHE_TYPE",
    "HostHardware",
    "SlotLease",
    "WorkerScalingProfile",
    "WorkerSlotGate",
    "build_scaling_profile",
    "detect_host_hardware",
]


if __name__ == "__main__":
    raise SystemExit(_cli())
