# HLPW6 AI/ML submission

Prepare, evaluate and package HiLiftAeroML predictions for the **6th High Lift Prediction Workshop (HLPW6) AI/ML Technical Working Group**.

**Preview v0.1.0:** synthetic tests pass. Workshop submissions remain closed until validation against the original evaluator and scoring data is complete. See [release status](docs/RELEASE_STATUS.md).

| Scope | Required predictions | Maximum score |
|---|---|---:|
| `surface_only` | Pressure and wall shear at every native surface point | 60/100 |
| `surface_and_volume` | Surface fields plus pressure and velocity at every valid native volume point | 100/100 |

Both scopes use the same 14 evaluation regimes. Missing volume metrics display **Not submitted** and earn zero points; weights stay fixed. Forces are integrated from surface fields. Profiles and regional diagnostics retain their native physical support.

## Start

Install Python 3.12 and `uv`, then:

```sh
git clone https://github.com/neilashton/hlpw6-aiml-submission.git
cd hlpw6-aiml-submission
uv sync --frozen
uv run hlpw6-aiml --help
uv run hlpw6-aiml list-splits
```

Follow the [participant guide](docs/PARTICIPANT_GUIDE.md) to declare training, write predictions, evaluate and deliver a submission ZIP. Its [synthetic example](docs/PARTICIPANT_GUIDE.md#offline-smoke-check) needs no dataset download.

Delivery is manual through a confidential HLPW6 channel, yet to be announced. Organisers review results privately; there is no automatic upload or public leaderboard.

## Reference

- [Scoring definitions](docs/SCORING.md)
- [Native support specification](docs/NATIVE_SUPPORT.md) for organiser integration
- [Development and source identities](docs/DEVELOPMENT.md), including the pinned dataset revision

Code: Apache-2.0. HiLiftAeroML data: CC-BY-4.0. Attribution: [NOTICE](NOTICE) and [UPSTREAM.json](UPSTREAM.json).
