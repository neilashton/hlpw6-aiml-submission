from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import (
    CASE_PATTERN,
    ID_PATTERN,
    ContractError,
    case_set_sha256,
    official_splits,
    scope_domains,
)
from .jsonio import read_json, write_json
from .training import TRAINING_DECLARATION_KEYS, validate_training_declaration

REQUIRED = {
    "schema",
    "schema_version",
    "submission_id",
    "method_name",
    "contact_email",
    "split_id",
    "test_case_ids",
    "prediction_scope",
    *TRAINING_DECLARATION_KEYS,
}
OPTIONAL = {"train_case_ids", "validation_case_ids", "methodology", "prediction_artifact"}


def _cases(value: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ContractError(f"{name} must be a non-empty list of case IDs")
    if any(not isinstance(v, str) or not CASE_PATTERN.fullmatch(v) for v in value):
        raise ContractError(f"{name} contains an invalid HiLiftAeroML case ID")
    if len(set(value)) != len(value):
        raise ContractError(f"{name} contains duplicate cases")
    return value


def validate_entry(document: dict[str, Any]) -> dict[str, Any]:
    missing, extra = REQUIRED - document.keys(), document.keys() - REQUIRED - OPTIONAL
    if missing or extra:
        raise ContractError(f"entry keys differ: missing={sorted(missing)}, extra={sorted(extra)}")
    if document["schema"] != "hlpw6-aiml-entry-v1" or document["schema_version"] != 1:
        raise ContractError("entry is not an HLPW6 v1 entry")
    for key in ("submission_id", "split_id"):
        if not isinstance(document[key], str) or not ID_PATTERN.fullmatch(document[key]):
            raise ContractError(f"invalid {key}")
    for key in ("method_name", "contact_email"):
        value = document[key]
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise ContractError(f"{key} must be non-empty and at most 200 characters")
    if "@" not in document["contact_email"]:
        raise ContractError("contact_email must be an email address")
    scope_domains(document["prediction_scope"])
    validate_training_declaration(document)
    test_ids = _cases(document["test_case_ids"], "test_case_ids")
    splits = official_splits()
    if document["split_id"] in splits:
        if test_ids != splits[document["split_id"]]["case_ids"]:
            raise ContractError("test_case_ids must match the exact ordered official split")
        for key in ("train_case_ids", "validation_case_ids"):
            if key in document and document[key] != splits[document["split_id"]][key]:
                raise ContractError(f"{key} differs from the published training regime")
    else:
        universe = {case for split in splits.values() for case in split["case_ids"]}
        if set(test_ids) - universe:
            raise ContractError("custom evaluation cases must be in the pinned case universe")
        if not {"train_case_ids", "validation_case_ids"} <= document.keys():
            raise ContractError("custom splits require complete train and validation case lists")
    memberships = [set(test_ids)]
    for key in ("train_case_ids", "validation_case_ids"):
        if key in document:
            members = set(_cases(document[key], key, allow_empty=True))
            if any(members & previous for previous in memberships):
                raise ContractError("train, validation, and test cases must not overlap")
            memberships.append(members)
    if "methodology" in document and not isinstance(document["methodology"], dict):
        raise ContractError("methodology must be an object")
    return document


def load_entry(root: Path | str) -> dict[str, Any]:
    path = Path(root)
    return validate_entry(read_json(path if path.is_file() else path / "entry.json"))


def split_record(entry: dict[str, Any]) -> dict[str, Any]:
    official = official_splits().get(entry["split_id"])
    return {
        "split_id": entry["split_id"],
        "label": official["split_label"] if official else entry["split_id"],
        "official": official is not None,
        "test_case_ids": entry["test_case_ids"],
        "test_case_count": len(entry["test_case_ids"]),
        "case_set_sha256": case_set_sha256(entry["test_case_ids"]),
        "train_case_count": len(
            official["train_case_ids"] if official else entry["train_case_ids"]
        ),
        "validation_case_count": len(
            official["validation_case_ids"] if official else entry["validation_case_ids"]
        ),
        "training_case_set_sha256": case_set_sha256(
            official["train_case_ids"] if official else entry["train_case_ids"]
        ),
        "validation_case_set_sha256": case_set_sha256(
            official["validation_case_ids"] if official else entry["validation_case_ids"]
        ),
        "training_definition": "Use the dataset-published training regime for the selected split.",
    }


def init_entry(
    root: Path | str,
    *,
    split_id: str,
    scope: str,
    submission_id: str,
    method_name: str,
    contact_email: str,
) -> dict[str, Any]:
    if split_id not in official_splits():
        raise ContractError(
            "init-entry requires an official split; declare custom lists explicitly"
        )
    document = {
        "schema": "hlpw6-aiml-entry-v1",
        "schema_version": 1,
        "submission_id": submission_id,
        "method_name": method_name,
        "contact_email": contact_email,
        "split_id": split_id,
        "test_case_ids": official_splits()[split_id]["case_ids"],
        "prediction_scope": scope,
        "training_regime": "from_scratch",
        "target_data_used": "official_train",
        "external_pretraining": False,
        "pretraining_data": [],
    }
    validate_entry(document)
    write_json(Path(root) / "entry.json", document, exclusive=True)
    return document
