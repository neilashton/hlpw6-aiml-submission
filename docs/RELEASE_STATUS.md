# Release status: 0.1.0 preview

The repositories and local workflow are implemented. **Workshop intake is not yet activated.**

Available now:

- HLPW6 entry declarations, all 14 pinned split definitions and training lists.
- Surface-only and surface-plus-volume interfaces, native prediction chunk writer, prepared-support evaluation adapter, compact packages and shared verification.
- Fixed-weight scores, integrated loads, compact profile encoding, regional diagnostics, a participant HTML report and offline synthetic workflow checks.
- A private local dashboard with all 1,355 native CFD plotting cases.

The original HiLiftAeroML workflow calls external native evaluator modules and compact scoring-support files. Those files are not included in its GitHub repository and were not available in this checkout. The new prepared-support adapter has passed synthetic integration tests; it has **not** been validated on the native Table 5 runs. The public float32 plotting copies cannot reconstruct the authoritative scoring support.

Before participant intake, the organiser must complete the following work on the machine that holds the original evaluation assets:

1. Locate the native submission, native evaluator core and recipe helper directories used by the existing Table 5 workflow. Record their immutable source revisions and normalization conventions.
2. Supply the original compact scoring-support release and its source truth authority. The expected upstream manifest SHA-256 is `7f946510cf8f63d05adaff4c91980d74f29c45ed7209b5a8f7e604e521cb823b`. If a newer audited release is selected, update the contract and version together.
3. Integrate or export the native support to the documented adapter format. Confirm every raw point ID, float32 surface weight, volume validity mask, load reference and profile interpolation stencil against the original evaluator. Surface-only exports must not require native volume data.
4. Run at least two real cases in both scopes, vary prediction chunk order and size, and compare fields, exact Cd/Cl/CmPitch and compact Cp/velocity scores against the original evaluator. Then complete a selected official split and compare its aggregates with the source result. Check working storage and runtime on that machine; native archives are very large.
5. Publish the immutable scoring-support access instructions, release/index hashes and the verification evidence. Add `contract/native-support-release.json` only after those checks pass, using schema `hlpw6-native-support-release-v1`, `status: validated`, and `index_sha256`. Release a new evaluator version and pin the dashboard to it.
6. Set up and document the HLPW6 confidential delivery destination. This repository does not reuse an AutoCFD5 upload link or publish participant results.

The production evaluator and package verifier deliberately reject workshop use until step 5. `--demonstration` is a separate synthetic test mode, and the dashboard rejects its packages by default. Demonstration results are always labelled and unranked when explicitly enabled for developer tests.

Package hashes verify integrity and consistency with retained evidence. They are not signatures or proof of model training. Organisers retain the usual responsibility for submission provenance and scientific review.
