# Scoring reference

The evaluation **contract** is the versioned set of scoring definitions, splits and support identities in [`contract/`](../contract/). Each output binds its full hash. [`scoring.json`](../contract/scoring.json) records the exact metric IDs, equations, units, directions, aggregation and weighting rules.

## Score components

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

## Reduction and physical support

Compute field errors over each complete case, then average cases equally:

- Reduce additive statistics across the full case before computing its errors.
- Surface primary fields use published float32 nodal dual-area weights; volume primary fields weight valid points equally.
- Vector relative L2 uses the three-component squared norm with one nodal weight. Dimensional vector MAE/RMSE reduce per scalar component.
- Loads use exact ordered-fan integration, per-case reference values and the published sign and angle conventions.
- Cp profiles retain physical graphs and disconnected branches. Velocity profiles preserve all five stations and their gaps.
- Score the compact representation after decoding, so compression and displayed predictions agree.

Native points are points on the original CFD mesh. **Scoring support** contains the reference arrays, weights, geometry and interpolation data used by the evaluator. The [support specification](NATIVE_SUPPORT.md) defines their exact format; dashboard plotting data cannot replace them.

Return to the [participant workflow](PARTICIPANT_GUIDE.md).
