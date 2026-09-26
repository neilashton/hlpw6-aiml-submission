from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contracts import EVALUATOR_VERSION, SCOPES, official_splits
from .entry import init_entry, load_entry
from .evaluation import evaluate_entry
from .fetch import fetch_data
from .packaging import package_result, verify_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HLPW6 AI/ML participant workflow")
    parser.add_argument("--version", action="version", version=EVALUATOR_VERSION)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-splits", help="List the 14 pinned training and evaluation regimes")
    initialize = sub.add_parser("init-entry", help="Create a participant entry declaration")
    initialize.add_argument("root", type=Path)
    initialize.add_argument("--split", required=True, choices=sorted(official_splits()))
    initialize.add_argument("--scope", choices=SCOPES, required=True)
    initialize.add_argument("--submission-id", required=True)
    initialize.add_argument("--method-name", required=True)
    initialize.add_argument("--contact-email", required=True)
    validate = sub.add_parser("validate-entry", help="Check an entry declaration")
    validate.add_argument("entry", type=Path)
    fetch = sub.add_parser(
        "fetch-data", help="Download pinned native archives; use --dry-run first"
    )
    fetch.add_argument("entry", type=Path)
    fetch.add_argument("destination", type=Path)
    fetch.add_argument("--case", action="append", dest="cases")
    fetch.add_argument("--dry-run", action="store_true")
    evaluate = sub.add_parser("evaluate-entry", help="Evaluate all declared native predictions")
    evaluate.add_argument("entry", type=Path)
    evaluate.add_argument(
        "--support", type=Path, required=True,
        help="Directory containing organiser-provided scoring support",
    )
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--scratch", type=Path, help="Directory for temporary prediction files"
    )
    evaluate.add_argument(
        "--resume", action="store_true",
        help="Reuse completed cases when input identities are unchanged",
    )
    evaluate.add_argument(
        "--demonstration",
        action="store_true",
        help="Use synthetic support; output is never a workshop submission",
    )
    package = sub.add_parser("package", help="Build and verify a submission ZIP")
    package.add_argument("result", type=Path)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--allow-demonstration", action="store_true")
    verify = sub.add_parser("verify-package", help="Verify a packaged result")
    verify.add_argument("package", type=Path)
    verify.add_argument("--allow-demonstration", action="store_true")
    report = sub.add_parser("report", help="Write a standalone local HTML metric report")
    report.add_argument("result", type=Path)
    report.add_argument("--output", type=Path, required=True)
    demo = sub.add_parser(
        "demo",
        help="Create synthetic inputs for an offline workflow check",
    )
    demo.add_argument("root", type=Path)
    demo.add_argument("--scope", choices=SCOPES, default="surface_and_volume")
    args = parser.parse_args(argv)
    try:
        if args.command == "list-splits":
            result = [
                {
                    "split_id": key,
                    "train": len(value["train_case_ids"]),
                    "validation": len(value["validation_case_ids"]),
                    "test": value["case_count"],
                }
                for key, value in official_splits().items()
            ]
        elif args.command == "init-entry":
            result = init_entry(
                args.root,
                split_id=args.split,
                scope=args.scope,
                submission_id=args.submission_id,
                method_name=args.method_name,
                contact_email=args.contact_email,
            )
        elif args.command == "validate-entry":
            entry = load_entry(args.entry)
            result = {
                "valid": True,
                "submission_id": entry["submission_id"],
                "prediction_scope": entry["prediction_scope"],
            }
        elif args.command == "fetch-data":
            entry = load_entry(args.entry)
            cases = args.cases or entry["test_case_ids"]
            if set(cases) - set(entry["test_case_ids"]):
                raise ValueError("--case must belong to the entry's declared test split")
            result = fetch_data(
                cases, entry["prediction_scope"], args.destination, dry_run=args.dry_run
            )
        elif args.command == "evaluate-entry":
            result = evaluate_entry(
                args.entry,
                args.support,
                args.output,
                resume=args.resume,
                demonstration=args.demonstration,
                scratch_root=args.scratch,
            )
        elif args.command == "package":
            result = package_result(
                args.result, args.output, allow_demonstration=args.allow_demonstration
            )
        elif args.command == "verify-package":
            verified = verify_package(args.package, allow_demonstration=args.allow_demonstration)
            result = {
                "valid": True,
                "sha256": verified.package_sha256,
                "size_bytes": verified.size_bytes,
                "prediction_scope": verified.result["prediction_scope"],
                "evaluation_kind": verified.result["evaluation_kind"],
            }
        elif args.command == "report":
            from .report import write_report

            result = write_report(args.result, args.output)
        else:
            from .demo import create_demo

            result = create_demo(args.root, args.scope)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"HLPW6: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
