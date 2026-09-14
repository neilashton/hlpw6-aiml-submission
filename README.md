# HLPW6 AI/ML submission

Participant tools for the **6th High Lift Prediction Workshop AI/ML Technical Working Group**, based on **HiLiftAeroML**. This public repository contains the submission tools; participant results are reviewed in a separate private organiser dashboard.

**Version 0.1.0 is a preview.** The software workflow is implemented and tested with synthetic native meshes. Workshop intake awaits access to and validation against the original native evaluator and authoritative scoring support. [Release status and remaining integration work](docs/RELEASE_STATUS.md).

| Submission scope | Required predictions | Maximum fixed-weight score |
|---|---|---:|
| `surface_only` | Complete native surface pressure and wall shear | 60/100 |
| `surface_and_volume` | Surface fields plus all valid native volume pressure and velocity | 100/100 |

Both scopes use the same 14 evaluation regimes and HiLift scientific definitions. Missing volume metrics are **Not submitted** and contribute zero points; weights are never renormalized. Forces come from the surface fields. Cp profiles, velocity profiles and regional diagnostics retain their native physical support.

## Start

```sh
git clone https://github.com/neilashton/hlpw6-aiml-submission.git
cd hlpw6-aiml-submission
uv sync --frozen
uv run hlpw6-aiml --help
uv run hlpw6-aiml list-splits
```

Read the [participant guide](docs/PARTICIPANT_GUIDE.md) for entries, pretrained model declarations, prediction writing, local evaluation, reports and compact ZIP delivery. The [native support specification](docs/NATIVE_SUPPORT.md) documents the pending organiser integration. No submission is uploaded automatically. The HLPW6 confidential delivery destination must be announced before intake.

## Offline verification

```sh
uv sync --frozen --extra test
uv run pytest
uv run ruff check src tests
uv run hlpw6-aiml demo ./var/demo --scope surface_only
uv run hlpw6-aiml evaluate-entry ./var/demo/entry --support ./var/demo/support \
  --output ./var/demo/result --demonstration
uv run hlpw6-aiml package ./var/demo/result --output ./var/demo.zip --allow-demonstration
```

The demo is synthetic and excluded from workshop intake. Use a new demo folder and `surface_and_volume` to check both domains.

## Source and data identities

The dataset is [NVIDIA HiLiftAeroML](https://huggingface.co/datasets/nvidia/HiLiftAeroML/tree/bbec30bcfc6103309c1375c5228b3ad0a586bfaf), pinned to revision `bbec30bcfc6103309c1375c5228b3ad0a586bfaf`. Its native archives retain their earlier immutable identities. Train/validation membership matches the dataset release; evaluation order preserves the inherited workflow. Each output binds the full HLPW6 contract hash.

The workflow derives from the AutoCFD5 repositories, with HiLift numerical reference routines from the HiLiftAeroML implementation in FluidsBench. These are retained for attribution in [NOTICE](NOTICE) and [UPSTREAM.json](UPSTREAM.json); participant-facing branding and package envelopes use HLPW6. Code is Apache-2.0. HiLiftAeroML data remains under its own CC-BY-4.0 license. No existing participant results or upload destinations were copied into this repository.
