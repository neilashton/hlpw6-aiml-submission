from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .jsonio import canonical_json_bytes, read_json, sha256_file

EVALUATOR_VERSION = "hlpw6-aiml-evaluator-v0.1.0"
DATASET_REPOSITORY = "nvidia/HiLiftAeroML"
DATASET_REVISION = "bbec30bcfc6103309c1375c5228b3ad0a586bfaf"
ARCHIVE_REVISION = "1c266d3869bc2968ff97d2107c9c3919be03ed32"
SCOPES = ("surface_only", "surface_and_volume")
CASE_PATTERN = re.compile(r"geo_LHC[0-9]{3}_AoA_[0-9]+\Z")
ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}\Z")
SURFACE_ONLY_UNAVAILABLE = frozenset(
    {"volume_velocity_rel_l2", "volume_pressure_rel_l2", "velocity_profile_r2"}
)


class ContractError(ValueError):
    """An input is inconsistent with the HLPW6 evaluation contract."""


def contract_root() -> Path:
    packaged = Path(__file__).parent / "contract"
    return packaged if packaged.is_dir() else Path(__file__).resolve().parents[2] / "contract"


def scope_domains(scope: str) -> tuple[str, ...]:
    if scope not in SCOPES:
        raise ContractError(f"prediction_scope must be one of {', '.join(SCOPES)}")
    return ("surface",) if scope == "surface_only" else ("surface", "volume")


def require_native_release() -> dict[str, Any]:
    path = contract_root() / "native-support-release.json"
    if not path.is_file():
        raise ContractError(
            "Workshop evaluation is not activated: the audited native evaluator and scoring-support release must be installed and validated first. See docs/RELEASE_STATUS.md."
        )
    release = read_json(path)
    if (
        release.get("schema") != "hlpw6-native-support-release-v1"
        or release.get("status") != "validated"
    ):
        raise ContractError("native support release has not passed scientific activation checks")
    return release


@lru_cache(maxsize=1)
def scoring() -> dict[str, Any]:
    return read_json(contract_root() / "scoring.json")


@lru_cache(maxsize=1)
def official_splits() -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted((contract_root() / "splits").glob("*.json")):
        item = read_json(path)
        ids = item["case_ids"]
        if len(ids) != item["case_count"] or len(ids) != len(set(ids)):
            raise ContractError(f"invalid case coverage in {path.name}")
        digest = case_set_sha256(ids)
        if digest != item["case_set_sha256"]:
            raise ContractError(f"split case identity differs in {path.name}")
        result[item["split_id"]] = item
    return result


def case_set_sha256(ids: list[str]) -> str:
    return hashlib.sha256("".join(f"{case_id}\n" for case_id in ids).encode()).hexdigest()


@lru_cache(maxsize=1)
def contract_identity() -> str:
    root = contract_root()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    inventory = {p.relative_to(root).as_posix(): sha256_file(p) for p in files}
    return hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()


def dataset_identity() -> dict[str, str]:
    return {"id": "hiliftaeroml", "repository": DATASET_REPOSITORY, "revision": DATASET_REVISION}


@lru_cache(maxsize=2)
def case_identities(kind: str) -> dict[str, Any]:
    filename = (
        "profile-support-identities.json"
        if kind == "profiles"
        else "native-support-identities.json"
    )
    return read_json(contract_root() / filename)["cases"]
