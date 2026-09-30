#!/usr/bin/env python3
"""Dependency-based reader and corpus annotation audit.

Examples:
    python -m src check
    python -m src sentence "PROTEIN0 binds PROTEIN1."
    python -m src run aimed full
    python -m src reproduce statistics --output DIR
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relation pattern and entity boundary audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode")

    p_run = sub.add_parser(
        "run",
        help="Read a corpus and export its evidence and audit workbook",
    )
    p_run.add_argument("dataset", nargs="?", default="aimed",
                         help="Dataset name (default: aimed)")
    p_run.add_argument("split", nargs="?", default="full",
                         help="Split (default: full)")
    p_run.add_argument("-n", "--limit", type=int, default=-1, help="Max sentences (-1 = all)")
    p_run.add_argument("--sample", type=int, default=None, dest="sample_size",
                         help="Read a reproducible sample of N sentences; output is marked as sampled.")
    p_run.add_argument("--sample-seed", type=int, default=42, dest="sample_seed_value",
                         help="Seed for --sample (default: 42; same seed -> same sentences)")
    p_run.add_argument("--stanza-package", default="craft", dest="stanza_package",
                         help="Stanza biomedical package (default: craft)")
    p_run.add_argument("--no-coref", action="store_true", help="Disable coreference resolution")
    p_run.add_argument("--no-nominal", action="store_true", help="Disable nominalization detection")
    p_run.add_argument("--no-cache", action="store_true", help="Disable native Stanza ParsedSentence cache")
    p_run.add_argument("--cpu", action="store_true", help="Disable parser GPU use")
    p_run.add_argument("--no-mine-paths", action="store_true",
                         help="Omit lexicalized-path review evidence (not recommended for final audit runs)")
    p_run.add_argument("--no-observations", action="store_true",
                         help="Omit additive mention graph, event observation, and pair trace sidecars "
                              "(not recommended for final audit runs)")
    p_run.add_argument("--force", action="store_true", help="Force rerun ignoring registry cache")

    p_sent = sub.add_parser("sentence", help="Inspect a single sentence through the pipeline")
    p_sent.add_argument("text", help="Sentence with PROTEIN0, PROTEIN1, ... markers")
    p_sent.add_argument("--stanza-package", default="craft", dest="stanza_package",
                        help="Stanza biomedical package (default: craft)")
    p_sent.add_argument("--no-coref", action="store_true", help="Disable coreference resolution")
    p_sent.add_argument("--no-nominal", action="store_true", help="Disable nominalization detection")
    p_sent.add_argument("--cpu", action="store_true", help="Disable parser GPU use")

    p_reproduce = sub.add_parser("reproduce", help="Verify or rebuild the paper experiment")
    p_reproduce.add_argument("level", choices=("verify", "statistics", "full"))
    p_reproduce.add_argument("--output", type=str, help="New output directory; frozen paper inputs are never overwritten")

    p_check = sub.add_parser("check", help="Check paper materials and saved results")
    p_check.add_argument("--parser", action="store_true", help="Also test the native reader on CPU")

    args = parser.parse_args()
    if not args.mode:
        parser.print_help()
        sys.exit(1)

    if args.mode == "check":
        from src.tools.smoke import check_saved_results, check_parser
        try:
            check_saved_results()
            if args.parser:
                check_parser()
        except RuntimeError as exc:
            parser.exit(1, f"FAIL: {exc}\n")
        return

    if args.mode == "reproduce":
        from pathlib import Path
        from src.reproduction.runner import reproduce
        try:
            reproduce(args.level, Path(args.output) if args.output else None)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            parser.exit(1, f"Reproduction failed: {exc}\n")
        return

    from src.pipeline import RunConfig, run_dataset, run_sentence

    if args.mode == "run":
        run_dataset(RunConfig(
            dataset=args.dataset, split=args.split,
            limit=args.limit,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed_value,
            use_coref=not args.no_coref,
            use_nominalization=not args.no_nominal,
            use_parser_cache=not args.no_cache,
            use_gpu=not args.cpu,
            stanza_package=args.stanza_package,
            mine_lexicalized_paths=not args.no_mine_paths,
            emit_event_observations=not args.no_observations,
            force_rerun=args.force,
        ))
    elif args.mode == "sentence":
        run_sentence(
            args.text,
            RunConfig(
                use_coref=not args.no_coref,
                use_nominalization=not args.no_nominal,
                use_gpu=not args.cpu,
                stanza_package=args.stanza_package,
            ), discovery=True,
        )


if __name__ == "__main__":
    main()
