from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except Exception:
            return [item.strip().strip('"').strip("'") for item in value[1:-1].split(",") if item.strip()]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = _parse_scalar(raw_value)
        parent[key] = value
        if isinstance(value, dict) and raw_value.strip() == "":
            stack.append((indent, value))
    return root


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
    except Exception:
        loaded = _minimal_yaml_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {source}")
    loaded["_config_path"] = str(source)
    return loaded


def dump_config(config: dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(config, sort_keys=False)
    except Exception:
        return _dump_minimal_yaml(config)


def _dump_minimal_yaml(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, child in value.items():
            if isinstance(child, dict):
                lines.append(f"{pad}{key}:")
                lines.append(_dump_minimal_yaml(child, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_format_scalar(child)}")
        return "\n".join(lines)
    return f"{pad}{_format_scalar(value)}"


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    if value is None:
        return "null"
    text = str(value)
    if any(ch in text for ch in [":", "#", "[", "]", "{", "}"]) or text == "":
        return repr(text)
    return text


def deep_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def deep_set(config: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    result = copy.deepcopy(config)
    current: dict[str, Any] = result
    parts = path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
    return result


def resolve_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(deep_get(config, "output.root_dir", "res"))
    run_dir = Path(deep_get(config, "output.run_dir", str(root / "doctamper_resnet34_h95_35epochs_comparison")))
    if not run_dir.is_absolute():
        run_dir = root / run_dir if run_dir.parts[:1] != (root.name,) else run_dir
    failure_dir = Path(deep_get(config, "failure_analysis.output_dir", str(root / "failure_case_analysis")))
    if not failure_dir.is_absolute():
        failure_dir = root / failure_dir if failure_dir.parts[:1] != (root.name,) else failure_dir
    tamper_dir = Path(deep_get(config, "tampering_type_analysis.output_dir", str(root / "tampering_type_analysis")))
    if not tamper_dir.is_absolute():
        tamper_dir = root / tamper_dir if tamper_dir.parts[:1] != (root.name,) else tamper_dir
    return {"root": root, "run": run_dir, "failure": failure_dir, "tamper": tamper_dir}
