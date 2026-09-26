# Development and source identities

## Check the software

```sh
uv sync --frozen --extra test
uv run pytest
uv run ruff check src tests
```

The [offline smoke check](PARTICIPANT_GUIDE.md#offline-smoke-check) exercises the complete synthetic workflow in either scope. Synthetic checks do not establish scientific equivalence: [release status](RELEASE_STATUS.md) lists the original-evaluator validation required before workshop intake.

## Dataset and contract

The dataset is [NVIDIA HiLiftAeroML](https://huggingface.co/datasets/nvidia/HiLiftAeroML/tree/bbec30bcfc6103309c1375c5228b3ad0a586bfaf), pinned to revision `bbec30bcfc6103309c1375c5228b3ad0a586bfaf`. Its native archives retain their earlier immutable identities. Train/validation membership matches the dataset release; evaluation order preserves the inherited workflow. Each output binds the full HLPW6 contract hash.

The workflow derives from the AutoCFD5 repositories, with HiLift numerical reference routines from the HiLiftAeroML implementation in FluidsBench. These are retained for attribution in [NOTICE](../NOTICE) and [UPSTREAM.json](../UPSTREAM.json); participant-facing branding and package envelopes use HLPW6. Code is Apache-2.0. HiLiftAeroML data remains under its own CC-BY-4.0 license. No existing participant results or upload destinations were copied into this repository.

The contract is the versioned set of evaluation definitions in [`contract/`](../contract/). See the [scoring reference](SCORING.md) for metric conventions and the [native support specification](NATIVE_SUPPORT.md) for the adapter format.
