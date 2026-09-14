"""A model-independent writer for the evaluator's native prediction interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import ContractError
from .jsonio import sha256_file, write_json


class PredictionWriter:
    def __init__(self, root: Path | str, domain: str):
        if domain not in {"surface", "volume"}:
            raise ContractError("prediction domain must be surface or volume")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.domain = domain
        self.chunks = []
        self.closed = False

    def write_chunk(self, raw_point_ids: np.ndarray, **fields: np.ndarray) -> None:
        if self.closed:
            raise ContractError("prediction manifest has already been finalized")
        expected = {"pressure", "wall_shear" if self.domain == "surface" else "velocity"}
        raw = np.asarray(raw_point_ids)
        if (
            set(fields) != expected
            or raw.ndim != 1
            or raw.dtype != np.dtype("int64")
            or not len(raw)
        ):
            raise ContractError(
                "write_chunk requires int64 raw IDs and the domain's exact field set"
            )
        arrays = {"raw_point_ids": raw}
        for key, value in fields.items():
            value = np.asarray(value)
            if value.shape != ((len(raw),) if key == "pressure" else (len(raw), 3)):
                raise ContractError("field shape must match raw IDs and its scalar/vector basis")
            if value.dtype.kind != "f" or not np.all(np.isfinite(value)):
                raise ContractError("predictions must be finite floating-point arrays")
            arrays[key] = value
        chunk = {}
        folder = self.root / f"chunk-{len(self.chunks):06d}"
        folder.mkdir()
        for key, value in arrays.items():
            path = folder / f"{key}.npy"
            with path.open("xb") as stream:
                np.save(stream, value, allow_pickle=False)
            chunk[key] = {
                "path": path.relative_to(self.root).as_posix(),
                "sha256": sha256_file(path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        self.chunks.append(chunk)

    def finish(self) -> Path:
        if self.closed or not self.chunks:
            raise ContractError("prediction writer is empty or already finalized")
        path = self.root / "manifest.json"
        write_json(
            path,
            {
                "schema": "hlpw6-native-predictions-v1",
                "domain": self.domain,
                "basis": "hiliftaeroml_nondimensional",
                "chunks": self.chunks,
            },
            exclusive=True,
        )
        self.closed = True
        return path
