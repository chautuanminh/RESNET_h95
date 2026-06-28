from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import deep_get
from .utils import write_csv, write_text

GB = 1024**3


def resolve_runtime(config: dict[str, Any]) -> dict[str, Any]:
    requested_device = str(deep_get(config, "gpu.device", "cuda")).lower()
    amp_enabled = bool(deep_get(config, "gpu.amp", True))
    requested_dtype = str(deep_get(config, "gpu.amp_dtype", "auto")).lower()
    torch_compile_setting = deep_get(config, "gpu.torch_compile", False)
    runtime = {
        "device": "cpu",
        "gpu_name": "none",
        "amp_dtype": "float32",
        "use_grad_scaler": False,
        "amp": False,
        "torch_compile": bool(torch_compile_setting is True or str(torch_compile_setting).lower() == "true"),
        "torch_compile_mode": str(deep_get(config, "gpu.torch_compile_mode", "reduce-overhead")),
        "channels_last": bool(deep_get(config, "gpu.channels_last", True)),
        "tf32": False,
        "cudnn_benchmark": bool(deep_get(config, "gpu.cudnn_benchmark", True)),
        "total_memory_gb": 0.0,
    }
    try:
        import torch  # type: ignore

        if requested_device == "cuda" and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            runtime["device"] = "cuda"
            runtime["gpu_name"] = name
            runtime["total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / GB, 4)
            if amp_enabled:
                runtime["amp"] = True
                if requested_dtype in {"bf16", "bfloat16"}:
                    runtime["amp_dtype"] = "bfloat16"
                elif requested_dtype in {"fp16", "float16"}:
                    runtime["amp_dtype"] = "float16"
                elif requested_dtype in {"fp32", "float32", "none", "false"}:
                    runtime["amp"] = False
                    runtime["amp_dtype"] = "float32"
                else:
                    runtime["amp_dtype"] = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
                runtime["use_grad_scaler"] = runtime["amp_dtype"] == "float16"
            tf32 = bool(deep_get(config, "gpu.tf32", True))
            runtime["tf32"] = tf32
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
            torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
            try:
                if tf32:
                    torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    except Exception:
        pass
    return runtime


def write_gpu_profile(path: str | Path, runtime: dict[str, Any]) -> Path:
    lines = ["# GPU Profile", ""]
    for key, value in runtime.items():
        lines.append(f"- {key}: {value}")
    return write_text(path, "\n".join(lines) + "\n")


def amp_dtype_for_torch(torch_module, runtime: dict[str, Any]):
    dtype = runtime.get("amp_dtype")
    if dtype == "bfloat16":
        return torch_module.bfloat16
    if dtype == "float16":
        return torch_module.float16
    return torch_module.float32


def cuda_memory_snapshot(torch_module) -> dict[str, float]:
    if not torch_module.cuda.is_available():
        return {
            "gpu_allocated_gb": 0.0,
            "gpu_reserved_gb": 0.0,
            "gpu_max_reserved_gb": 0.0,
            "gpu_max_reserved_percent": 0.0,
        }
    total = max(1, int(torch_module.cuda.get_device_properties(0).total_memory))
    max_reserved = int(torch_module.cuda.max_memory_reserved())
    return {
        "gpu_allocated_gb": round(torch_module.cuda.memory_allocated() / GB, 4),
        "gpu_reserved_gb": round(torch_module.cuda.memory_reserved() / GB, 4),
        "gpu_max_reserved_gb": round(max_reserved / GB, 4),
        "gpu_max_reserved_percent": round(max_reserved / total * 100.0, 4),
    }


def autotune_batch_size(
    config: dict[str, Any],
    runtime: dict[str, Any],
    path: str | Path,
    *,
    manual_batch_size: int | None = None,
    model_factory: Callable[[dict[str, Any]], Any] | None = None,
    criterion_factory: Callable[[], Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> int:
    configured_batch = int(deep_get(config, "training.batch_size", 4))
    if manual_batch_size is not None:
        selected = int(manual_batch_size)
        _write_autotune_rows(
            path,
            [
                _autotune_row(
                    selected,
                    "manual_override",
                    runtime,
                    selected=True,
                    note="CLI --batch-size bypassed autotuning",
                )
            ],
        )
        return selected

    if runtime.get("device") != "cuda" or not bool(deep_get(config, "gpu.auto_tune_batch_size", False)):
        _write_autotune_rows(
            path,
            [
                _autotune_row(
                    configured_batch,
                    "disabled",
                    runtime,
                    selected=True,
                    note="autotuning disabled or CUDA unavailable",
                )
            ],
        )
        return configured_batch

    if model_factory is None or criterion_factory is None:
        raise ValueError("GPU batch-size autotuning requires model_factory and criterion_factory")

    import torch  # type: ignore

    candidates = [int(value) for value in deep_get(config, "gpu.batch_size_candidates", [configured_batch])]
    candidates = sorted({candidate for candidate in candidates if candidate > 0})
    if not candidates:
        candidates = [configured_batch]
    memory_fraction_limit = float(deep_get(config, "gpu.auto_tune_memory_fraction", 0.86))
    image_size = int(deep_get(config, "preprocessing.image_size", 512))
    in_channels = int(deep_get(config, "model.in_channels", 2))
    classes = int(deep_get(config, "model.classes", 1))
    device = torch.device("cuda")
    amp_dtype = amp_dtype_for_torch(torch, runtime)
    autocast_enabled = bool(runtime.get("amp")) and runtime.get("amp_dtype") != "float32"

    if log:
        log(f"Batch autotune: probing candidates {candidates} at [{in_channels}, {image_size}, {image_size}]")
    model = model_factory(config).to(device)
    if runtime.get("channels_last"):
        model = model.to(memory_format=torch.channels_last)
    model.train()
    criterion = criterion_factory()
    rows: list[dict[str, Any]] = []
    safe_candidates: list[int] = []
    total_memory = max(1, int(torch.cuda.get_device_properties(0).total_memory))

    for candidate in candidates:
        peak_reserved = 0
        status = "pass"
        note = ""
        x = y = logits = loss = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.zero_grad(set_to_none=True)
            x = torch.randn((candidate, in_channels, image_size, image_size), device=device, dtype=torch.float32)
            y = torch.randint(
                0,
                2,
                (candidate, classes, image_size, image_size),
                device=device,
                dtype=torch.float32,
            )
            if runtime.get("channels_last"):
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            peak_reserved = int(torch.cuda.max_memory_reserved())
            memory_fraction = peak_reserved / total_memory
            if memory_fraction <= memory_fraction_limit:
                safe_candidates.append(candidate)
            else:
                status = "over_limit"
                note = f"reserved memory exceeded {memory_fraction_limit:.2f} fraction"
        except torch.cuda.OutOfMemoryError:
            status = "oom"
            note = "CUDA out of memory during probe"
            peak_reserved = int(torch.cuda.max_memory_reserved())
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                status = "oom"
                note = "CUDA out of memory during probe"
                peak_reserved = int(torch.cuda.max_memory_reserved())
            else:
                status = "fail"
                note = str(exc).replace("\n", " ")[:240]
        finally:
            model.zero_grad(set_to_none=True)
            del x, y, logits, loss
            torch.cuda.empty_cache()
        rows.append(
            _autotune_row(
                candidate,
                status,
                runtime,
                peak_reserved_gb=peak_reserved / GB if peak_reserved else 0.0,
                memory_fraction=peak_reserved / total_memory if peak_reserved else 0.0,
                note=note,
            )
        )
        if log:
            log(f"Batch autotune: {candidate} -> {status}, peak_reserved={peak_reserved / GB:.2f} GB")

    selected = max(safe_candidates) if safe_candidates else candidates[0]
    if not safe_candidates and log:
        log(f"Batch autotune: no candidate stayed below {memory_fraction_limit:.2f}; using smallest candidate {selected}")
    for row in rows:
        row["selected"] = "yes" if int(row["candidate_batch_size"]) == selected else "no"
    _write_autotune_rows(path, rows)
    del model
    torch.cuda.empty_cache()
    return selected


def _autotune_row(
    candidate: int,
    status: str,
    runtime: dict[str, Any],
    *,
    selected: bool = False,
    peak_reserved_gb: float = 0.0,
    memory_fraction: float = 0.0,
    note: str = "",
) -> dict[str, Any]:
    return {
        "candidate_batch_size": int(candidate),
        "status": status,
        "device": runtime.get("device"),
        "gpu_name": runtime.get("gpu_name"),
        "amp_dtype": runtime.get("amp_dtype"),
        "peak_reserved_gb": round(float(peak_reserved_gb), 4),
        "peak_reserved_percent": round(float(memory_fraction) * 100.0, 4),
        "memory_fraction": round(float(memory_fraction), 6),
        "selected": "yes" if selected else "no",
        "note": note,
    }


def _write_autotune_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    return write_csv(
        path,
        rows,
        [
            "candidate_batch_size",
            "status",
            "device",
            "gpu_name",
            "amp_dtype",
            "peak_reserved_gb",
            "peak_reserved_percent",
            "memory_fraction",
            "selected",
            "note",
        ],
    )
