from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .contracts import (
    ARCHIVE_REVISION,
    DATASET_REPOSITORY,
    ContractError,
    contract_root,
    scope_domains,
)
from .jsonio import read_json, sha256_file, write_json


def download_file(url: str, path: Path, *, sha256: str, size_bytes: int | None = None) -> Path:
    if path.is_file() and not path.is_symlink():
        if (size_bytes is None or path.stat().st_size == size_bytes) and sha256_file(
            path
        ) == sha256:
            return path
        raise ContractError(f"existing file differs from its immutable identity: {path.name}")
    if path.exists() or path.is_symlink():
        raise ContractError("download destination is not a regular file")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    total, digest = 0, hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "hlpw6-aiml/1.0"})
        with os.fdopen(fd, "wb") as target, urllib.request.urlopen(request, timeout=120) as source:
            while block := source.read(8 * 1024 * 1024):
                total += len(block)
                if size_bytes is not None and total > size_bytes:
                    raise ContractError("download exceeds its pinned size")
                target.write(block)
                digest.update(block)
        if digest.hexdigest() != sha256 or (size_bytes is not None and total != size_bytes):
            raise ContractError("download checksum or size differs from the pinned release")
        os.replace(temporary, path)
        return path
    finally:
        temporary.unlink(missing_ok=True)


def data_plan(case_ids: list[str], scope: str) -> list[dict[str, Any]]:
    records = read_json(
        contract_root() / "source-identity/hiliftaeroml-public-source-identity-v1.json"
    )
    registry = {item["case_id"]: item for item in records["cases"]}
    files = []
    for case_id in case_ids:
        if case_id not in registry:
            raise ContractError(f"case is outside the pinned dataset registry: {case_id}")
        for domain in scope_domains(scope):
            record = registry[case_id][domain]
            relative = record["archive"]["repository_path"]
            files.append(
                {
                    "case_id": case_id,
                    "domain": domain,
                    "path": relative,
                    "url": f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/{ARCHIVE_REVISION}/{quote(relative)}",
                    "size_bytes": record["archive"]["size_bytes"],
                    "sha256": record["archive"]["lfs_sha256"],
                    "member": record["member"],
                }
            )
    return files


def fetch_data(
    case_ids: list[str], scope: str, destination: Path, *, dry_run: bool = False
) -> dict[str, Any]:
    plan = data_plan(case_ids, scope)
    summary = {
        "prediction_scope": scope,
        "case_count": len(case_ids),
        "file_count": len(plan),
        "download_bytes": sum(row["size_bytes"] for row in plan),
        "extracted_bytes": sum(row["member"]["declared_size_bytes"] for row in plan),
        "files": [
            {key: row[key] for key in ("case_id", "domain", "path", "size_bytes", "sha256")}
            for row in plan
        ],
    }
    if dry_run:
        return summary
    destination.mkdir(parents=True, exist_ok=True)
    for row in plan:
        download_file(
            row["url"],
            destination / row["path"],
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
        )
    return summary


def extract_native_archive(
    archive: Path, destination: Path, record: dict[str, Any]
) -> dict[str, Any]:
    """Extract exactly the one pinned regular VTU, retaining an archive-to-content receipt."""
    if (
        archive.is_symlink()
        or archive.stat().st_size != record["size_bytes"]
        or sha256_file(archive) != record["sha256"]
    ):
        raise ContractError("native archive does not match its pinned source identity")
    member = record["member"]
    target = destination / member["path"]
    receipt_path = destination / (member["path"] + ".source.json")
    if target.is_symlink() or receipt_path.is_symlink():
        raise ContractError("native extraction cannot replace or follow a symlink")
    if target.exists():
        receipt = read_json(receipt_path)
        if (
            receipt["archive_sha256"] == record["sha256"]
            and sha256_file(target) == receipt["content_sha256"]
        ):
            return receipt
        raise ContractError("existing native content has no matching source receipt")
    destination.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".native-", dir=destination)
    temporary = Path(name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as target_stream, tarfile.open(archive, "r|gz") as bundle:
            item = bundle.next()
            if (
                item is None
                or not item.isfile()
                or item.name != member["path"]
                or "/" in item.name
                or item.size != member["declared_size_bytes"]
            ):
                raise ContractError(
                    "native archive does not contain exactly the pinned regular member"
                )
            stream = bundle.extractfile(item)
            if stream is None:
                raise ContractError("native archive member could not be read")
            total = 0
            with stream:
                while block := stream.read(8 * 1024 * 1024):
                    total += len(block)
                    if total > item.size:
                        raise ContractError("native archive member exceeds its declared size")
                    target_stream.write(block)
                    digest.update(block)
            if total != item.size or bundle.next() is not None:
                raise ContractError("native archive is truncated or has extra members")
        os.replace(temporary, target)
        receipt = {
            "schema": "hlpw6-native-source-receipt-v1",
            "archive_sha256": record["sha256"],
            "content_sha256": digest.hexdigest(),
            "size_bytes": target.stat().st_size,
        }
        write_json(receipt_path, receipt, exclusive=True)
        return receipt
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
