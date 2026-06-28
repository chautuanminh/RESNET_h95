from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from .config import deep_get, deep_set, dump_config, load_config, resolve_output_paths
from .datasets import discover_indices, export_dataset_index, make_data_loader, make_dataset
from .gpu import amp_dtype_for_torch, autotune_batch_size, cuda_memory_snapshot, resolve_runtime, write_gpu_profile
from .losses import BCEDiceLoss
from .metrics import StreamingBinaryMetrics
from .models import create_model, forward_shape, write_model_summary
from .report import write_output_contract, write_pdf_report, write_protocol_documents, write_res_summary, write_root_plan
from .splits import TRAINING_DATASET, create_or_load_split, write_leakage_report
from .tamper_types import assign_tamper_type, load_tamper_metadata
from .utils import ensure_writable_dir, write_csv, write_json, write_text
from .visualization import save_training_curves


TRAIN_METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_f1",
    "val_f1",
    "train_images_per_sec",
    "val_images_per_sec",
    "gpu_allocated_gb",
    "gpu_reserved_gb",
    "gpu_max_reserved_gb",
    "gpu_max_reserved_percent",
    "batch_size",
    "effective_batch_size",
    "amp_dtype",
    "lr",
]


class TrainingLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_writable_dir(self.path.parent, purpose="training log")
        try:
            self._file = self.path.open("w", encoding="utf-8")
        except PermissionError as exc:
            raise PermissionError(
                "Could not open training log for writing: "
                f"{self.path}. Parent directory: {self.path.parent}. "
                "Check ownership and write permissions with: "
                f"whoami; ls -ld {self.path.parent}; ls -l {self.path}. "
                "Prefer a fresh writable output.root_dir under runs/. "
                "If you own the directory or file, use chmod u+w on that path."
            ) from exc

    def __call__(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def run_train(config_path: str | Path, resume: str | None = None, batch_size: int | None = None) -> Path:
    config = load_config(config_path)
    paths = resolve_output_paths(config)
    root = ensure_writable_dir(paths["root"], purpose="output root")
    run_dir = ensure_writable_dir(paths["run"], purpose="training run output")
    ensure_writable_dir(run_dir / "checkpoints", purpose="checkpoint output")
    logger = TrainingLogger(run_dir / "training.log")
    try:
        logger(f"Training setup: config={config_path}")
        write_protocol_documents(root)
        write_root_plan(root)
        write_output_contract(root)
        write_pdf_report(root, "Training initialized; final metrics are written after evaluation.")
        write_res_summary(root, [])
        write_text(run_dir / "config_snapshot.yaml", dump_config(config))

        runtime = resolve_runtime(config)
        write_gpu_profile(run_dir / "gpu_profile.txt", runtime)
        logger(
            "Runtime: "
            f"device={runtime.get('device')} gpu={runtime.get('gpu_name')} "
            f"amp={runtime.get('amp')} amp_dtype={runtime.get('amp_dtype')} "
            f"channels_last={runtime.get('channels_last')} torch_compile={runtime.get('torch_compile')}"
        )

        dummy_mode = bool(deep_get(config, "runtime.dummy", False) or deep_get(config, "model.force_dummy", False))
        autotune_runtime = {**runtime, "device": "cpu"} if dummy_mode else runtime
        selected_batch_size = autotune_batch_size(
            config,
            autotune_runtime,
            run_dir / "batch_size_autotune.csv",
            manual_batch_size=batch_size,
            model_factory=None if dummy_mode else create_model,
            criterion_factory=None if dummy_mode else BCEDiceLoss,
            log=logger,
        )
        resolved_config = deep_set(config, "training.batch_size", int(selected_batch_size))
        write_text(run_dir / "config_resolved.yaml", dump_config(resolved_config))
        logger(f"Selected batch size: {selected_batch_size}")

        train_folder = deep_get(resolved_config, "data.train_sets", [TRAINING_DATASET])[0]
        all_indices = discover_indices(resolved_config, train_folder)
        split = create_or_load_split(
            all_indices,
            run_dir,
            seed=int(deep_get(resolved_config, "split.seed", 42)),
            val_count=int(deep_get(resolved_config, "split.val_count", 10000)),
            source_folder=train_folder,
        )
        logger(f"Split: train={len(split.train_indices)} val={len(split.val_indices)}")

        test_counts = {
            label: len(discover_indices(resolved_config, folder))
            for label, folder in dict(deep_get(resolved_config, "data.test_sets", {})).items()
        }
        write_leakage_report(
            root / "leakage_check_report.md",
            len(split.train_indices),
            len(split.val_indices),
            test_counts,
            train_val_overlap=0,
        )

        summary_model = create_model(resolved_config)
        write_model_summary(root / "model_summary.txt", summary_model, resolved_config)
        _assert_model_contract(summary_model, resolved_config, runtime, logger)
        del summary_model

        index_rows = _dataset_index_rows(resolved_config, split.train_indices, split.val_indices)
        export_dataset_index(run_dir / "dataset_index.csv", index_rows)

        if dummy_mode:
            _write_dummy_training_outputs(resolved_config, run_dir, resume, selected_batch_size, runtime, logger)
        else:
            _torch_train(resolved_config, split.train_indices, split.val_indices, run_dir, selected_batch_size, resume, logger)
        logger("Training finished.")
        return run_dir / "checkpoints" / "best_model.pth"
    finally:
        logger.close()


def _dataset_index_rows(config: dict[str, Any], train_indices: list[int], val_indices: list[int]) -> list[dict[str, Any]]:
    metadata = load_tamper_metadata(deep_get(config, "data.tamper_metadata_dir", "tampering_types"))
    rows: list[dict[str, Any]] = []
    train_folder = deep_get(config, "data.train_sets", [TRAINING_DATASET])[0]
    for split_name, folder, indices in [
        ("train", train_folder, train_indices),
        ("val", train_folder, val_indices),
    ]:
        for idx in indices:
            assignment = assign_tamper_type(folder, idx, metadata, enable_heuristic=False)
            rows.append(
                {
                    "split": split_name,
                    "folder": folder,
                    "dataset_index": idx,
                    "image_id": f"{split_name}_{idx:06d}",
                    "tamper_type": assignment.tamper_type,
                    "tamper_type_source": assignment.tamper_type_source,
                    "tamper_type_confidence": assignment.tamper_type_confidence,
                    "tamper_type_reason": assignment.tamper_type_reason,
                }
            )
    for label, folder in dict(deep_get(config, "data.test_sets", {})).items():
        for idx in discover_indices(config, folder):
            assignment = assign_tamper_type(label, idx, metadata, enable_heuristic=False)
            rows.append(
                {
                    "split": label,
                    "folder": folder,
                    "dataset_index": idx,
                    "image_id": f"{label}_{idx:06d}",
                    "tamper_type": assignment.tamper_type,
                    "tamper_type_source": assignment.tamper_type_source,
                    "tamper_type_confidence": assignment.tamper_type_confidence,
                    "tamper_type_reason": assignment.tamper_type_reason,
                }
            )
    return rows


def _assert_model_contract(model, config: dict[str, Any], runtime: dict[str, Any], log: TrainingLogger) -> None:
    image_size = int(deep_get(config, "preprocessing.image_size", 512))
    in_channels = int(deep_get(config, "model.in_channels", 2))
    classes = int(deep_get(config, "model.classes", 1))
    batch = 2
    expected_input = [batch, in_channels, image_size, image_size]
    expected_logits = [batch, classes, image_size, image_size]
    if model.__class__.__name__ == "DummyResNetUNet":
        import numpy as np  # type: ignore

        logits_shape = forward_shape(model, np.zeros(expected_input, dtype=np.float32))
    else:
        import torch  # type: ignore

        device = torch.device(runtime["device"])
        model = model.to(device)
        if runtime["device"] == "cuda" and runtime.get("channels_last"):
            model = model.to(memory_format=torch.channels_last)
        model.eval()
        with torch.inference_mode():
            x = torch.zeros(expected_input, device=device, dtype=torch.float32)
            if runtime["device"] == "cuda" and runtime.get("channels_last"):
                x = x.contiguous(memory_format=torch.channels_last)
            logits = model(x)
            logits_shape = [int(value) for value in logits.shape]
        del x, logits
        if runtime["device"] == "cuda":
            torch.cuda.empty_cache()
    if logits_shape != expected_logits:
        raise AssertionError(f"Dry run shape mismatch: input={expected_input}, logits={logits_shape}")
    log(f"Dry run OK: input={expected_input} logits={expected_logits}")


def _write_dummy_training_outputs(
    config: dict[str, Any],
    run_dir: Path,
    resume: str | None,
    batch_size: int,
    runtime: dict[str, Any],
    log: TrainingLogger,
) -> None:
    epochs = int(deep_get(config, "training.epochs", 1))
    grad_accum = max(1, int(deep_get(config, "training.gradient_accumulation_steps", 1)))
    rows = []
    val_rows = []
    for epoch in range(1, epochs + 1):
        train_loss = round(1.0 / (epoch + 1), 6)
        val_loss = round(1.0 / (epoch + 1.5), 6)
        train_f1 = round(epoch / max(1, epochs), 6)
        val_f1 = train_f1
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "train_images_per_sec": 0,
                "val_images_per_sec": 0,
                "gpu_allocated_gb": 0,
                "gpu_reserved_gb": 0,
                "gpu_max_reserved_gb": 0,
                "gpu_max_reserved_percent": 0,
                "batch_size": batch_size,
                "effective_batch_size": batch_size * grad_accum,
                "amp_dtype": runtime.get("amp_dtype", "float32"),
                "lr": deep_get(config, "training.lr", 0.0001),
            }
        )
        val_rows.append(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "val_iou": val_f1,
                "val_precision": val_f1,
                "val_recall": val_f1,
                "val_images_per_sec": 0,
                "checkpoint_selection_source": "internal_validation_only",
            }
        )
        log(f"Epoch {epoch}/{epochs}: dummy train_loss={train_loss:.4f} val_f1={val_f1:.4f}")
    write_csv(run_dir / "train_metrics.csv", rows, TRAIN_METRIC_FIELDS)
    write_csv(run_dir / "val_metrics.csv", val_rows)
    save_training_curves(run_dir / "plots" / "training_curves.png", rows, val_rows)
    payload = {
        "backend": "dummy",
        "checkpoint_selection_source": "internal_validation_only",
        "epoch": epochs,
        "best_score": rows[-1]["val_f1"] if rows else -1,
        "resume": resume or "",
        "config": config,
    }
    write_json(run_dir / "checkpoints" / "last_checkpoint.pth", payload)
    write_json(run_dir / "checkpoints" / "best_model.pth", payload)


def _torch_train(
    config: dict[str, Any],
    train_indices: list[int],
    val_indices: list[int],
    run_dir: Path,
    batch_size: int,
    resume: str | None,
    log: TrainingLogger,
) -> None:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    runtime = resolve_runtime(config)
    device = torch.device(runtime["device"])
    model = create_model(config).to(device)
    if runtime["device"] == "cuda" and runtime.get("channels_last"):
        model = model.to(memory_format=torch.channels_last)

    start_epoch = 1
    best_f1 = -1.0
    resume_payload: dict[str, Any] = {}
    if resume:
        resume_payload = torch.load(resume, map_location="cpu")
        state = resume_payload.get("model_state", resume_payload)
        state = {key.replace("_orig_mod.", ""): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        best_f1 = float(resume_payload.get("best_score", -1.0))
        log(f"Resumed checkpoint: {resume} start_epoch={start_epoch} best_f1={best_f1:.4f}")

    if runtime.get("torch_compile") and hasattr(torch, "compile"):
        log(f"Compiling model: mode={runtime.get('torch_compile_mode')}")
        model = torch.compile(model, mode=str(runtime.get("torch_compile_mode", "reduce-overhead")))

    optimizer = _create_optimizer(config, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(deep_get(config, "training.epochs", 35)),
        eta_min=float(deep_get(config, "training.eta_min", 1e-6)),
    )
    criterion = BCEDiceLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=bool(runtime.get("use_grad_scaler") and runtime["device"] == "cuda"))
    if resume_payload:
        optimizer.load_state_dict(resume_payload.get("optimizer_state", optimizer.state_dict()))
        scheduler.load_state_dict(resume_payload.get("scheduler_state", scheduler.state_dict()))
        if resume_payload.get("scaler_state"):
            scaler.load_state_dict(resume_payload["scaler_state"])

    train_folder = deep_get(config, "data.train_sets", [TRAINING_DATASET])[0]
    train_dataset = make_dataset(config, "train", train_folder, train_indices)
    val_dataset = make_dataset(config, "val", train_folder, val_indices)
    train_drop_last = bool(deep_get(config, "training.drop_last", True)) and len(train_dataset) >= batch_size
    train_loader = _build_loader(config, train_dataset, batch_size, shuffle=True, drop_last=train_drop_last)
    val_loader = _build_loader(config, val_dataset, batch_size, shuffle=False, drop_last=False)

    grad_accum = max(1, int(deep_get(config, "training.gradient_accumulation_steps", 1)))
    effective_batch = batch_size * grad_accum
    amp_dtype = amp_dtype_for_torch(torch, runtime)
    autocast_enabled = bool(runtime.get("amp") and runtime["device"] == "cuda" and runtime.get("amp_dtype") != "float32")
    log_interval = max(1, int(deep_get(config, "training.log_interval_batches", 50)))
    total_epochs = int(deep_get(config, "training.epochs", 35))
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    log(
        "Training loop: "
        f"epochs={total_epochs} batch_size={batch_size} effective_batch_size={effective_batch} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
    )

    for epoch in range(start_epoch, total_epochs + 1):
        if runtime["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_stream = StreamingBinaryMetrics(0.5)
        running_loss = 0.0
        batches = 0
        images = 0
        epoch_start = time.perf_counter()
        log(f"Epoch {epoch}/{total_epochs}: training start")

        for step, batch in enumerate(train_loader, start=1):
            try:
                x, y = _batch_to_tensors(batch, device, runtime)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
                    logits = model(x)
                    raw_loss = criterion(logits, y)
                    loss = raw_loss / grad_accum
                scaler.scale(loss).backward()
                should_step = step % grad_accum == 0 or step == len(train_loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    clip_norm = float(deep_get(config, "training.gradient_clip_norm", 1.0))
                    if clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                batch_size_actual = int(x.shape[0])
                running_loss += float(raw_loss.detach().cpu().item())
                batches += 1
                images += batch_size_actual
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                labels = y.detach().cpu().numpy()
                train_stream.update(probs, labels)
                if step % log_interval == 0 or step == len(train_loader):
                    elapsed = max(1e-8, time.perf_counter() - epoch_start)
                    log(
                        f"Epoch {epoch}/{total_epochs}: batch {step}/{len(train_loader)} "
                        f"loss={running_loss / max(1, batches):.4f} ips={images / elapsed:.2f}"
                    )
                del x, y, logits, raw_loss, loss, probs, labels
            except torch.cuda.OutOfMemoryError as exc:
                _handle_cuda_oom(exc, config, run_dir, model, optimizer, scheduler, scaler, best_f1, epoch, batch_size, log)

        scheduler.step()
        train_elapsed = max(1e-8, time.perf_counter() - epoch_start)
        train_metrics = train_stream.metrics()
        val_metrics = _torch_validate(model, val_loader, device, runtime, amp_dtype, criterion)
        memory = cuda_memory_snapshot(torch)
        current_lr = scheduler.get_last_lr()[0]
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batches),
            "val_loss": val_metrics["val_loss"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["val_f1"],
            "train_images_per_sec": images / train_elapsed,
            "val_images_per_sec": val_metrics["val_images_per_sec"],
            **memory,
            "batch_size": batch_size,
            "effective_batch_size": effective_batch,
            "amp_dtype": runtime.get("amp_dtype"),
            "lr": current_lr,
        }
        train_rows.append(row)
        val_rows.append({**val_metrics, "epoch": epoch, "checkpoint_selection_source": "internal_validation_only"})

        checkpoint_best = max(best_f1, val_metrics["val_f1"])
        payload = _checkpoint_payload(epoch, model, optimizer, scheduler, scaler, checkpoint_best, config)
        torch.save(payload, run_dir / "checkpoints" / "last_checkpoint.pth")
        if val_metrics["val_f1"] >= best_f1:
            best_f1 = val_metrics["val_f1"]
            payload["best_score"] = best_f1
            torch.save(payload, run_dir / "checkpoints" / "best_model.pth")
            checkpoint_msg = "best checkpoint updated"
        else:
            checkpoint_msg = "last checkpoint updated"
        write_csv(run_dir / "train_metrics.csv", train_rows, TRAIN_METRIC_FIELDS)
        write_csv(run_dir / "val_metrics.csv", val_rows)
        save_training_curves(run_dir / "plots" / "training_curves.png", train_rows, val_rows)
        log(
            f"Epoch {epoch}/{total_epochs}: train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} train_f1={row['train_f1']:.4f} "
            f"val_f1={row['val_f1']:.4f} {checkpoint_msg}"
        )


def _build_loader(config: dict[str, Any], dataset, batch_size: int, *, shuffle: bool, drop_last: bool):
    workers = int(deep_get(config, "training.num_workers", 0))
    return make_data_loader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(deep_get(config, "training.pin_memory", True)),
        persistent_workers=bool(deep_get(config, "training.persistent_workers", workers > 0)),
        prefetch_factor=deep_get(config, "training.prefetch_factor", 2) if workers > 0 else None,
        drop_last=drop_last,
    )


def _create_optimizer(config: dict[str, Any], model):
    import torch  # type: ignore

    name = str(deep_get(config, "training.optimizer", "adamw")).lower()
    lr = float(deep_get(config, "training.lr", 1e-4))
    weight_decay = float(deep_get(config, "training.weight_decay", 1e-4))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _batch_to_tensors(batch: list[dict[str, Any]], device, runtime: dict[str, Any]):
    import numpy as np  # type: ignore
    import torch  # type: ignore

    non_blocking = runtime.get("device") == "cuda"
    x = torch.from_numpy(np.stack([sample["input"] for sample in batch])).to(device, non_blocking=non_blocking)
    y = torch.from_numpy(np.stack([sample["mask"] for sample in batch])).to(device, non_blocking=non_blocking)
    if runtime["device"] == "cuda" and runtime.get("channels_last"):
        x = x.contiguous(memory_format=torch.channels_last)
    return x, y


def _torch_validate(model, loader, device, runtime: dict[str, Any], amp_dtype, criterion) -> dict[str, float]:
    import torch  # type: ignore

    model.eval()
    stream = StreamingBinaryMetrics(0.5)
    running_loss = 0.0
    batches = 0
    images = 0
    start = time.perf_counter()
    autocast_enabled = bool(runtime.get("amp") and runtime["device"] == "cuda" and runtime.get("amp_dtype") != "float32")
    with torch.inference_mode():
        for batch in loader:
            x, y = _batch_to_tensors(batch, device, runtime)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            running_loss += float(loss.detach().cpu().item())
            batches += 1
            images += int(x.shape[0])
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels = y.detach().cpu().numpy()
            stream.update(probs, labels)
            del x, y, logits, loss, probs, labels
    elapsed = max(1e-8, time.perf_counter() - start)
    metrics = stream.metrics()
    return {
        "val_loss": running_loss / max(1, batches),
        "val_f1": metrics["f1"],
        "val_iou": metrics["iou"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_images_per_sec": images / elapsed,
    }


def _checkpoint_payload(epoch: int, model, optimizer, scheduler, scaler, best_score: float, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_score": best_score,
        "config": config,
    }


def _handle_cuda_oom(
    exc: BaseException,
    config: dict[str, Any],
    run_dir: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    best_f1: float,
    epoch: int,
    batch_size: int,
    log: TrainingLogger,
) -> None:
    import torch  # type: ignore

    optimizer.zero_grad(set_to_none=True)
    scaler_states = getattr(scaler, "_per_optimizer_states", None)
    if scaler_states is not None:
        scaler_states.clear()
    torch.cuda.empty_cache()
    checkpoint = run_dir / "checkpoints" / "last_checkpoint.pth"
    try:
        torch.save(_checkpoint_payload(max(0, epoch - 1), model, optimizer, scheduler, scaler, best_f1, config), checkpoint)
        saved = f"saved {checkpoint}"
    except Exception as save_exc:
        saved = f"could not save checkpoint: {save_exc}"
    smaller = max(1, int(batch_size) // 2)
    resume_command = (
        f"python -m src.train --config {config.get('_config_path', 'configs/resnet_h95_config.yaml')} "
        f"--resume {checkpoint} --batch-size {smaller}"
    )
    log(f"CUDA OOM at epoch {epoch}: {exc}")
    log(f"OOM recovery: {saved}")
    log(f"Resume with a smaller batch: {resume_command}")
    raise RuntimeError(f"CUDA OOM during training. Resume suggestion: {resume_command}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)
    run_train(args.config, resume=args.resume, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
