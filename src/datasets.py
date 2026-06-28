from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

from .config import deep_get
from .preprocessing import preprocess_gray_h95
from .tamper_types import assign_tamper_type, load_tamper_metadata
from .utils import ensure_dir, write_csv


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    folder: str
    count: int


class SyntheticDocTamperDataset:
    def __init__(
        self,
        split_label: str,
        folder: str,
        indices: list[int],
        image_size: int = 64,
        metadata_root: str | Path = "tampering_types",
    ) -> None:
        self.split_label = split_label
        self.folder = folder
        self.indices = list(indices)
        self.image_size = image_size
        self.metadata = load_tamper_metadata(metadata_root)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        np = _np()
        idx = int(self.indices[item])
        height = width = self.image_size
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        image = (210 + ((x_grid + idx * 3) % 30)).astype("uint8")
        for line in range(6, height - 6, 10):
            image[line : line + 2, 8 : width - 8] = 60 + (idx % 20)
        mask = np.zeros((height, width), dtype="uint8")
        x0 = 8 + (idx * 7) % max(8, width // 2)
        y0 = 8 + (idx * 5) % max(8, height // 2)
        x1 = min(width - 4, x0 + 8 + idx % 8)
        y1 = min(height - 4, y0 + 5 + idx % 6)
        mask[y0:y1, x0:x1] = 255
        image[y0:y1, x0:x1] = (image[y0:y1, x0:x1] // 2 + 40).astype("uint8")
        processed = preprocess_gray_h95(image, mask, size=(self.image_size, self.image_size))
        assignment = assign_tamper_type(
            self.split_label,
            idx,
            self.metadata,
            enable_heuristic=True,
            heuristic_features={
                "mask_area_ratio": float(mask.sum() / 255 / (height * width)),
                "component_count": 1,
                "patch_similarity": 0.6 + (idx % 5) * 0.06,
                "h95_inside_outside_ratio": 1.0 + (idx % 4) * 0.15,
            },
        )
        return {
            "dataset_index": idx,
            "image_id": f"{self.split_label}_{idx:06d}",
            "test_set": self.split_label,
            "folder": self.folder,
            "image": image,
            "mask_raw": mask,
            "input": processed.input,
            "mask": processed.mask,
            "gray": processed.gray,
            "h95": processed.h95,
            "tamper_type": assignment.tamper_type,
            "tamper_type_source": assignment.tamper_type_source,
            "tamper_type_confidence": assignment.tamper_type_confidence,
            "tamper_type_reason": assignment.tamper_type_reason,
        }


class DocTamperLMDBDataset:
    def __init__(
        self,
        root: str | Path,
        split_label: str,
        folder: str,
        indices: list[int],
        image_size: int = 512,
        metadata_root: str | Path = "tampering_types",
    ) -> None:
        self.root = Path(root)
        self.split_label = split_label
        self.folder = folder
        self.indices = list(indices)
        self.image_size = image_size
        self.metadata = load_tamper_metadata(metadata_root)
        self._env = None

    def __len__(self) -> int:
        return len(self.indices)

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._env = None

    def close(self) -> None:
        env = self._env
        self._env = None
        if env is not None:
            env.close()

    def _open(self):
        if self._env is None:
            try:
                import lmdb  # type: ignore
            except Exception as exc:
                raise ImportError("LMDB dataset loading requires the lmdb package") from exc
            path = self.root / self.folder
            if not path.exists():
                raise FileNotFoundError(f"DocTamper folder not found: {path}")
            try:
                self._env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=64)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to open DocTamper LMDB in read-only shared mode: "
                    f"{path}. The reader uses lock=False so it can coexist with an already open LMDB environment. "
                    "Check that the path is a valid LMDB directory and that the current user has read permission."
                ) from exc
        return self._env

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = int(self.indices[item])
        image_bytes, mask_bytes = self._read_lmdb_pair(idx)
        image = _decode_image(image_bytes)
        mask = _decode_image(mask_bytes)
        processed = preprocess_gray_h95(image, mask, size=(self.image_size, self.image_size))
        assignment = assign_tamper_type(self.split_label, idx, self.metadata, enable_heuristic=False)
        return {
            "dataset_index": idx,
            "image_id": f"{self.split_label}_{idx:06d}",
            "test_set": self.split_label,
            "folder": self.folder,
            "image": image,
            "mask_raw": mask,
            "input": processed.input,
            "mask": processed.mask,
            "gray": processed.gray,
            "h95": processed.h95,
            "tamper_type": assignment.tamper_type,
            "tamper_type_source": assignment.tamper_type_source,
            "tamper_type_confidence": assignment.tamper_type_confidence,
            "tamper_type_reason": assignment.tamper_type_reason,
        }

    def _read_lmdb_pair(self, idx: int) -> tuple[bytes, bytes]:
        env = self._open()
        image_candidates, mask_candidates = _key_candidates(idx)
        sample_candidates = _sample_key_candidates(idx)
        with env.begin(write=False) as txn:
            sample_value = _first_value(txn, sample_candidates)
            if sample_value is not None:
                parsed = _parse_sample_value(sample_value)
                if parsed is not None:
                    return parsed
            image_bytes = _first_value(txn, image_candidates)
            mask_bytes = _first_value(txn, mask_candidates)
            if image_bytes is None or mask_bytes is None:
                sampled = []
                cursor = txn.cursor()
                for i, (key, _) in enumerate(cursor):
                    if i >= 20:
                        break
                    sampled.append(key.decode("utf-8", errors="replace"))
                raise KeyError(
                    f"Unable to find image/mask keys for index {idx}. Sampled LMDB keys: {sampled}"
                )
            return image_bytes, mask_bytes


def _key_candidates(idx: int) -> tuple[list[bytes], list[bytes]]:
    forms = [str(idx), f"{idx:06d}", f"{idx:08d}", f"{idx:09d}"]
    image_prefixes = ["image", "img", "input", "tampered"]
    mask_prefixes = ["mask", "label", "gt", "target"]
    image = []
    mask = []
    for form in forms:
        image.append(form.encode())
        for prefix in image_prefixes:
            image.extend(
                [
                    f"{prefix}-{form}".encode(),
                    f"{prefix}_{form}".encode(),
                    f"{prefix}/{form}".encode(),
                ]
            )
        for prefix in mask_prefixes:
            mask.extend(
                [
                    f"{prefix}-{form}".encode(),
                    f"{prefix}_{form}".encode(),
                    f"{prefix}/{form}".encode(),
                ]
            )
    return image, mask


def _sample_key_candidates(idx: int) -> list[bytes]:
    forms = [str(idx), f"{idx:06d}", f"{idx:08d}", f"{idx:09d}"]
    candidates = []
    for form in forms:
        candidates.extend(
            [
                form.encode(),
                f"sample-{form}".encode(),
                f"sample_{form}".encode(),
                f"data-{form}".encode(),
                f"data_{form}".encode(),
            ]
        )
    return candidates


def _parse_sample_value(payload: bytes) -> tuple[bytes, bytes] | None:
    import pickle

    try:
        value = pickle.loads(payload)
    except Exception:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2 and isinstance(value[0], bytes) and isinstance(value[1], bytes):
        return value[0], value[1]
    if isinstance(value, dict):
        image = None
        mask = None
        for key in ["image", "img", "jpg", "jpeg", "png", "tampered"]:
            if isinstance(value.get(key), bytes):
                image = value[key]
                break
        for key in ["mask", "label", "gt", "target"]:
            if isinstance(value.get(key), bytes):
                mask = value[key]
                break
        if image is not None and mask is not None:
            return image, mask
    return None


def _first_value(txn, keys: list[bytes]) -> bytes | None:
    for key in keys:
        value = txn.get(key)
        if value is not None:
            return value
    return None


def _decode_image(payload: bytes):
    try:
        from PIL import Image  # type: ignore

        return Image.open(BytesIO(payload)).convert("L")
    except Exception as exc:
        raise ValueError("Unable to decode image bytes from LMDB") from exc


def _np():
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise ImportError("Synthetic dataset requires numpy") from exc
    return np


def discover_indices(config: dict[str, Any], folder: str) -> list[int]:
    if bool(deep_get(config, "data.synthetic.enabled", False)):
        counts = deep_get(config, "data.synthetic.counts", {})
        short = folder.replace("DocTamperV1-", "")
        count = int(counts.get(folder, counts.get(short, deep_get(config, "data.synthetic.default_count", 8))))
        return list(range(count))

    metadata_root = Path(deep_get(config, "data.tamper_metadata_dir", "tampering_types"))
    metadata = load_tamper_metadata(metadata_root)
    for alias in [folder, folder.replace("DocTamperV1-", "")]:
        if alias in metadata:
            return sorted(metadata[alias])

    root = Path(deep_get(config, "data.root", "/storage/student7/cmt/data/archive"))
    dataset_path = root / folder
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found and no metadata indices available: {dataset_path}")
    try:
        import lmdb  # type: ignore

        env = lmdb.open(str(dataset_path), readonly=True, lock=False, readahead=False, max_readers=64)
        try:
            with env.begin(write=False) as txn:
                length_raw = txn.get(b"num-samples") or txn.get(b"length") or txn.get(b"__len__")
                if length_raw:
                    return list(range(int(length_raw.decode("utf-8"))))
                indices = set()
                for key, _ in txn.cursor():
                    digits = bytes(ch for ch in key if 48 <= ch <= 57)
                    if digits:
                        indices.add(int(digits))
                if indices:
                    return sorted(indices)
        finally:
            env.close()
    except Exception as exc:
        raise RuntimeError(
            "Unable to discover LMDB indices in read-only shared mode for "
            f"{dataset_path}: {exc}"
        ) from exc
    raise RuntimeError(f"Unable to discover any indices for {dataset_path}")


def make_dataset(config: dict[str, Any], split_label: str, folder: str, indices: list[int]):
    image_size = int(deep_get(config, "preprocessing.image_size", 512))
    metadata_root = deep_get(config, "data.tamper_metadata_dir", "tampering_types")
    if bool(deep_get(config, "data.synthetic.enabled", False)):
        return SyntheticDocTamperDataset(split_label, folder, indices, image_size=image_size, metadata_root=metadata_root)
    return DocTamperLMDBDataset(
        deep_get(config, "data.root", "/storage/student7/cmt/data/archive"),
        split_label,
        folder,
        indices,
        image_size=image_size,
        metadata_root=metadata_root,
    )


def iter_batches(dataset, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for i in range(len(dataset)):
        batch.append(dataset[i])
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def collate_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(samples)


def make_data_loader(
    dataset,
    batch_size: int,
    *,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    drop_last: bool = False,
    generator=None,
):
    try:
        from torch.utils.data import DataLoader  # type: ignore
    except Exception as exc:
        raise ImportError("PyTorch DataLoader requires the torch package") from exc

    workers = max(0, int(num_workers))
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": workers,
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
        "collate_fn": collate_samples,
    }
    if generator is not None:
        kwargs["generator"] = generator
    if workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def export_dataset_index(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    return write_csv(
        path,
        rows,
        [
            "split",
            "folder",
            "dataset_index",
            "image_id",
            "tamper_type",
            "tamper_type_source",
            "tamper_type_confidence",
            "tamper_type_reason",
        ],
    )
