from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class JsonContractError(ValueError):
    """Raised when a retained JSON document is malformed or changes during use."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JsonContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise JsonContractError(f"JSON input must be a regular file: {source}")
    return read_json_bytes(source.read_bytes())


def read_json_bytes(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                JsonContractError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise JsonContractError(f"invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise JsonContractError("JSON document must be an object")
    return value


def write_json(path: Path | str, value: object, *, exclusive: bool = False) -> dict[str, object]:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise JsonContractError("JSON output must use the .json suffix")
    if exclusive and (destination.exists() or destination.is_symlink()):
        raise JsonContractError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "file": destination.name,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }
