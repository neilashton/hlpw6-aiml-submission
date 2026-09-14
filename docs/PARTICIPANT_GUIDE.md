# HLPW6 AI/ML participant guide

This workflow is for the **6th High Lift Prediction Workshop AI/ML Technical Working Group** using **HiLiftAeroML**. Version 0.1.0 is a preview; read [release status](RELEASE_STATUS.md) before planning a workshop run.

## Install and choose a split

Install Python 3.12 and `uv`, then:

```sh
git clone https://github.com/neilashton/hlpw6-aiml-submission.git
cd hlpw6-aiml-submission
uv sync --frozen
uv run hlpw6-aiml list-splits
uv run hlpw6-aiml init-entry ./var/my-entry \
  --split full --scope surface_only --submission-id team-method-v1 \
  --method-name 'My model' --contact-email participant@example.org
```

Use `surface_and_volume` for both domains. Every entry evaluates the **complete test set** for one declared split. Training regimes remain distinct even when they share a test set. The exact evaluation order is the order in the entry; it is inherited from the reference workflow. Membership matches the dataset's published test sets, whose ordering differs. Full train and validation lists are pinned in `contract/splits/`.

The initial entry declares training from scratch on the official training data. Edit the declaration if using a pretrained model:

| `training_regime` | `target_data_used` | `external_pretraining` | `pretraining_data` |
|---|---|---|---|
| `from_scratch` | `official_train` | `false` | empty list |
| `pretrained_zero_shot` | `none` | `true` | named datasets |
| `pretrained_official_train` | `official_train` | `true` | named datasets |

Pretraining entries can be names or objects with `name` and an optional HTTP(S) `url`. Describe the model and training procedure in optional `methodology`. Custom split IDs require explicit train, validation and test lists with no overlap; their test cases must come from the pinned 1,355-case evaluation universe. Custom results are compared only with identical case and training definitions.

## Native predictions

Download only the domains needed by the declared scope. Inspect the size first:

```sh
uv run hlpw6-aiml fetch-data ./var/my-entry ./data --dry-run
# Optional one-case download, selecting an ID from entry.json:
uv run hlpw6-aiml fetch-data ./var/my-entry ./data --case geo_LHC001_AoA_20
```

Downloads use immutable repository paths, sizes and SHA-256 hashes. The command downloads native VTU archives; it does not prepare the organiser scoring support, run a model or automatically extract the multi-gigabyte datasets. Dataset metadata, mesh normalization and reference quantities must also follow the published HiLiftAeroML recipe.

Predict **all native surface points**, with no surface sampling. For volume, predict every valid native point, retaining the original raw VTU IDs. Raw Float32 `avg(P) == 0` identifies excluded volume rows. Do not renumber valid volume points.

| Domain | Array | Shape | Basis |
|---|---|---|---|
| Surface | `pressure` | N | `(P - p_inf) / q_inf` |
| Surface | `wall_shear` | N × 3 | `tau_wall / q_inf` |
| Volume | `pressure` | N | `(P - p_inf) / q_inf` |
| Volume | `velocity` | N × 3 | `U / abs(U_inf)` |

Predictions are finite floating-point arrays. Raw point IDs are int64. Write model outputs using the model-independent adapter:

```python
from hlpw6_aiml.predictions import PredictionWriter

writer = PredictionWriter(
    'var/my-entry/cases/geo_LHC001_AoA_20/surface', 'surface'
)
# Repeat for every chunk of model output. Arrays below come from your model.
writer.write_chunk(raw_point_ids, pressure=predicted_cp, wall_shear=predicted_cf)
writer.finish()
```

For volume use a separate `.../<case_id>/volume` writer with `pressure` and `velocity`. Chunk order is flexible; coverage must be exact, with no unknown, excluded or duplicate IDs. A surface-only entry needs no volume prediction directory. Do not submit independently fitted force coefficients: they are integrated from the surface fields.

## Evaluate, inspect and package

These workshop commands require the validated native support release described in the release status:

```sh
uv run hlpw6-aiml validate-entry ./var/my-entry
uv run hlpw6-aiml evaluate-entry ./var/my-entry \
  --support /path/to/validated-hlpw6-support --output ./output/my-entry
uv run hlpw6-aiml report ./output/my-entry --output ./var/my-report.html
uv run hlpw6-aiml package ./output/my-entry --output ./var/team-method-v1-full.zip
uv run hlpw6-aiml verify-package ./var/team-method-v1-full.zip
```

Use `--scratch /path/with-space` to place native prediction memory maps on suitable storage. `--resume` reuses complete cases only for unchanged entry, support-index and prediction identities. Retain the full output directory while resuming. Changed inputs require a fresh output directory.

The delivery ZIP contains the entry, aggregate scores, per-case evidence, compact predicted profiles and optional regional diagnostics. It excludes native fields, meshes, scoring truth, checkpoints and local working files. A SHA-256 sidecar accompanies the ZIP. Packaging is deterministic for identical result files.

Send the ZIP and checksum through the **HLPW6 organiser's confidential delivery channel once announced**. Keep model outputs and detailed local reports on your machine. Participant scores are reviewed in the private organiser dashboard; there is no public leaderboard or automatic upload. A delivery endpoint has not yet been configured for this preview.

## Scores and missing volume

Weights remain fixed at 50% fields, 25% forces and 25% profiles:

| Component | Weight | Transform |
|---|---:|---|
| Surface pressure relative L2 (%) | 15% | `max(0, 1-error/15)` |
| Surface wall shear relative L2 (%) | 10% | `max(0, 1-error/20)` |
| Volume velocity relative L2 (%) | 15% | `max(0, 1-error/12)` |
| Volume pressure relative L2 (%) | 10% | `max(0, 1-error/15)` |
| Cd R² | 15% | clamp to [0,1] |
| Cl R² | 10% | clamp to [0,1] |
| Velocity profile R² | 15% | clamp to [0,1] |
| Cp cut R² | 10% | clamp to [0,1] |

Each transformed component is multiplied by 100 before weighting. Missing volume metrics are `null` and display as **Not submitted**; their weighted contribution is zero. There is no renormalization: the surface-only maximum is **60/100**, and the surface-plus-volume maximum is **100/100**. Exact pitching moment, dimensional errors, equal-entity surface errors and regional diagnostics provide extra information with no additional score weight.

Field errors reduce additive statistics over each complete case, then macro-average cases equally. Surface primary fields use published float32 nodal dual-area weights. Volume primary fields use equal weight on valid points. Vector relative L2 uses the three-component squared norm with one nodal weight. Vector dimensional MAE/RMSE reduce per scalar component. Loads use exact ordered-fan integration, per-case reference values and the published sign and angle conventions. Cp profiles retain physical graphs and disconnected branches; velocity profiles preserve all five stations and their gaps. The compact representation is scored after decoding, so compression and displayed predictions agree.

## Offline smoke check

No dataset download is needed for this clearly labelled synthetic check:

```sh
uv run hlpw6-aiml demo ./var/demo --scope surface_only
uv run hlpw6-aiml evaluate-entry ./var/demo/entry --support ./var/demo/support \
  --output ./var/demo/result --demonstration
uv run hlpw6-aiml package ./var/demo/result --output ./var/demo.zip --allow-demonstration
uv run hlpw6-aiml verify-package ./var/demo.zip --allow-demonstration
```

Repeat in a new folder with `--scope surface_and_volume` to exercise volume. These fixtures are synthetic software checks and cannot be submitted as workshop results.
