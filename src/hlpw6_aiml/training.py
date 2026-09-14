from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

TRAINING_REGIME_FROM_SCRATCH = "from_scratch"
TRAINING_REGIME_PRETRAINED_ZERO_SHOT = "pretrained_zero_shot"
TRAINING_REGIME_PRETRAINED_OFFICIAL_TRAIN = "pretrained_official_train"
TRAINING_REGIMES = frozenset(
    {
        TRAINING_REGIME_FROM_SCRATCH,
        TRAINING_REGIME_PRETRAINED_ZERO_SHOT,
        TRAINING_REGIME_PRETRAINED_OFFICIAL_TRAIN,
    }
)
TARGET_DATA_NONE = "none"
TARGET_DATA_OFFICIAL_TRAIN = "official_train"
TARGET_DATA_OPTIONS = frozenset({TARGET_DATA_NONE, TARGET_DATA_OFFICIAL_TRAIN})
TRAINING_DECLARATION_KEYS = (
    "training_regime",
    "target_data_used",
    "external_pretraining",
    "pretraining_data",
)


class TrainingDeclarationError(ValueError):
    """Raised when model-training provenance is incomplete or inconsistent."""


def _text(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise TrainingDeclarationError(
            f"{label} must be a non-empty string of at most {maximum} characters"
        )
    return value.strip()


def validate_training_declaration(document: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in TRAINING_DECLARATION_KEYS if key not in document]
    if missing:
        raise TrainingDeclarationError("training declaration is missing: " + ", ".join(missing))

    regime = document.get("training_regime")
    target = document.get("target_data_used")
    external = document.get("external_pretraining")
    data = document.get("pretraining_data")
    if regime not in TRAINING_REGIMES:
        raise TrainingDeclarationError(
            "training_regime must be 'from_scratch', 'pretrained_zero_shot', "
            "or 'pretrained_official_train'"
        )
    if target not in TARGET_DATA_OPTIONS:
        raise TrainingDeclarationError("target_data_used must be 'none' or 'official_train'")
    if not isinstance(external, bool):
        raise TrainingDeclarationError("external_pretraining must be a boolean")
    if not isinstance(data, list):
        raise TrainingDeclarationError("pretraining_data must be an array")

    normalized_data: list[str | dict[str, str]] = []
    for index, item in enumerate(data):
        label = f"pretraining_data[{index}]"
        if isinstance(item, str):
            normalized_data.append(_text(item, label, 500))
            continue
        if not isinstance(item, Mapping) or set(item) - {"name", "url"} or "name" not in item:
            raise TrainingDeclarationError(
                f"{label} must be a name string or an object containing name and optional url"
            )
        normalized: dict[str, str] = {"name": _text(item.get("name"), f"{label}.name", 500)}
        if "url" in item:
            url = _text(item.get("url"), f"{label}.url", 2000)
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise TrainingDeclarationError(f"{label}.url must be an absolute HTTP(S) URL")
            normalized["url"] = url
        normalized_data.append(normalized)

    identities = [
        json.dumps(item, sort_keys=True, separators=(",", ":")) for item in normalized_data
    ]
    if len(identities) != len(set(identities)):
        raise TrainingDeclarationError("pretraining_data entries must be unique")

    if regime == TRAINING_REGIME_FROM_SCRATCH:
        if external is not False or normalized_data:
            raise TrainingDeclarationError(
                "from_scratch requires external_pretraining=false and empty pretraining_data"
            )
        if target != TARGET_DATA_OFFICIAL_TRAIN:
            raise TrainingDeclarationError(
                "from_scratch requires target_data_used='official_train'"
            )
    elif regime == TRAINING_REGIME_PRETRAINED_ZERO_SHOT:
        if external is not True or not normalized_data:
            raise TrainingDeclarationError(
                "pretrained_zero_shot requires external_pretraining=true and named pretraining_data"
            )
        if target != TARGET_DATA_NONE:
            raise TrainingDeclarationError("pretrained_zero_shot requires target_data_used='none'")
    elif external is not True or not normalized_data:
        raise TrainingDeclarationError(
            "pretrained_official_train requires external_pretraining=true and named pretraining_data"
        )
    elif target != TARGET_DATA_OFFICIAL_TRAIN:
        raise TrainingDeclarationError(
            "pretrained_official_train requires target_data_used='official_train'"
        )

    return {
        "training_regime": str(regime),
        "target_data_used": str(target),
        "external_pretraining": external,
        "pretraining_data": normalized_data,
    }


__all__ = [
    "TARGET_DATA_NONE",
    "TARGET_DATA_OFFICIAL_TRAIN",
    "TRAINING_DECLARATION_KEYS",
    "TRAINING_REGIME_FROM_SCRATCH",
    "TRAINING_REGIME_PRETRAINED_OFFICIAL_TRAIN",
    "TRAINING_REGIME_PRETRAINED_ZERO_SHOT",
    "TRAINING_REGIMES",
    "TrainingDeclarationError",
    "validate_training_declaration",
]
