"""Bounded ZIP reading shared by participant packaging and organiser import."""

from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts import ContractError
from .jsonio import canonical_json_bytes, read_json_bytes, sha256_file

MAXIMUM_MEMBER_BYTES = 128 * 1024 * 1024
MAXIMUM_TOTAL_BYTES = 2 * 1024**3
MAXIMUM_MEMBERS = 20_000


def safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or ":" in value
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in value.split("/"))
        or str(path) != value
    ):
        raise ContractError("archive member has an unsafe path")
    return value


def zip_inventory(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    items = archive.infolist()
    if len(items) > MAXIMUM_MEMBERS:
        raise ContractError("archive has too many members")
    members = {}
    total = 0
    for item in items:
        name = safe_relative_path(item.filename)
        mode = item.external_attr >> 16
        if name in members or item.is_dir() or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
            raise ContractError("archive has duplicate or non-regular members")
        if item.flag_bits & 1 or item.compress_type not in (
            zipfile.ZIP_STORED,
            zipfile.ZIP_DEFLATED,
        ):
            raise ContractError("archive uses encryption or an unsupported compression method")
        if item.file_size > MAXIMUM_MEMBER_BYTES or item.file_size < 0:
            raise ContractError("archive member exceeds the size limit")
        if item.file_size > 1024 * 1024 and item.file_size > 1000 * max(1, item.compress_size):
            raise ContractError("archive member has an excessive compression ratio")
        total += item.file_size
        members[name] = item
    if total > MAXIMUM_TOTAL_BYTES:
        raise ContractError("archive exceeds the expanded size limit")
    return members


def member_digest(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest, total = hashlib.sha256(), 0
    with archive.open(info) as stream:
        while data := stream.read(1024 * 1024):
            total += len(data)
            if total > info.file_size or total > MAXIMUM_MEMBER_BYTES:
                raise ContractError("archive member size differs from its directory")
            digest.update(data)
    if total != info.file_size:
        raise ContractError("archive member is truncated")
    return digest.hexdigest()


def verify_manifest(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members = zip_inventory(archive)
    if "package-manifest.json" not in members:
        raise ContractError("package manifest is missing")
    manifest = read_json_bytes(archive.read(members["package-manifest.json"]))
    if manifest.get("schema") != "hlpw6-aiml-package-manifest-v1":
        raise ContractError("not an HLPW6 package manifest")
    rows = manifest.get("files")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ContractError("package manifest files must be a list")
    names = [row.get("path") for row in rows]
    if not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ContractError("package manifest has invalid or duplicate members")
    if set(names) != set(members) - {"package-manifest.json"}:
        raise ContractError("manifest does not name exactly every package member")
    for row in rows:
        item = members[row["path"]]
        if row.get("size_bytes") != item.file_size or row.get("sha256") != member_digest(
            archive, item
        ):
            raise ContractError(f"package checksum or size differs: {item.filename}")
    return members


def deterministic_zip(root: Path, output: Path, paths: list[str]) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise ContractError("refusing to overwrite an existing delivery package")
    rows = []
    for relative in sorted(paths):
        safe_relative_path(relative)
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ContractError("package input must be a regular file inside the result directory")
        if path.stat().st_size > MAXIMUM_MEMBER_BYTES:
            raise ContractError("package member is too large")
        rows.append(
            {"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    manifest = canonical_json_bytes({"schema": "hlpw6-aiml-package-manifest-v1", "files": rows})
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted([*paths, "package-manifest.json"]):
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                if name == "package-manifest.json":
                    archive.writestr(info, manifest, compresslevel=9)
                else:
                    with (
                        (root / name).open("rb") as source,
                        archive.open(info, "w", force_zip64=True) as target,
                    ):
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
        with zipfile.ZipFile(output) as archive:
            verify_manifest(archive)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
    return {
        "path": str(output),
        "sha256": digest,
        "size_bytes": output.stat().st_size,
        "member_count": len(rows) + 1,
    }


def npz_arrays(payload: bytes, expected: set[str]) -> dict[str, Any]:
    """Reject object arrays and oversized NPY headers before NumPy allocates."""
    import numpy as np

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = zip_inventory(archive)
        if set(members) != {name + ".npy" for name in expected}:
            raise ContractError("prediction NPZ arrays differ from the declared scope")
        for info in members.values():
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ContractError("unsupported NPY format")
                if dtype.hasobject or fortran or len(shape) > 2:
                    raise ContractError("unsafe prediction array layout")
                count = 1
                for dimension in shape:
                    count *= dimension
                if (
                    count * dtype.itemsize > MAXIMUM_MEMBER_BYTES
                    or count * dtype.itemsize + stream.tell() != info.file_size
                ):
                    raise ContractError("NPY shape differs from its bounded member size")
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in expected}
