#!/usr/bin/env python3
"""Verify that a parser environment can emit the shared syntax contract."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def check_import(module: str, package: str | None = None) -> dict:
    package = package or module
    try:
        importlib.import_module(module)
        return {"name": module, "version": package_version(package), "ok": True}
    except Exception as exc:
        return {"name": module, "version": package_version(package), "ok": False, "error": str(exc)}


def parser_smoke(
    *,
    backend: str = "stanza",
    use_nominal: bool,
    use_coref: bool,
    gpu: bool,
) -> dict:
    try:
        from src.parsing import build_parser_backend
        parser = build_parser_backend(
            backend,
            use_nominalization=use_nominal,
            use_coref=use_coref,
            use_gpu=gpu,
        )
        parsed = parser.parse("PROTEIN0 phosphorylates PROTEIN1.", "__env_smoke__")
        payload = {
            "ok": True,
            "tokens": [n.text for n in parsed.nodes],
            "parser_name": parsed.parser_name,
        }
        return payload
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Skip heavy model smoke tests")
    ap.add_argument("--full", action="store_true", help="Run coref/parser smoke tests")
    ap.add_argument("--gpu", action="store_true", help="Ask Stanza to use GPU during parser smoke")
    args = ap.parse_args()

    full = args.full or (not args.quick)
    parser_imports = [
        check_import("stanza"),
        check_import("torch"),
        check_import("transformers"),
        check_import("lemminflect"),
        check_import("peft"),
        check_import("accelerate"),
    ]
    report = {
        "python": sys.version.split()[0],
        "project_root": str(PROJECT_ROOT),
        "imports": parser_imports + [
            check_import("openpyxl"),
            check_import("yaml", "PyYAML"),
            check_import("tqdm"),
        ],
        "parser_smoke": {"skipped": not full},
    }

    if full:
        report["parser_smoke"] = parser_smoke(
            backend="stanza",
            use_nominal=True,
            use_coref=True,
            gpu=args.gpu,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))

    failures = []
    failures.extend(item["name"] for item in report["imports"] if not item.get("ok"))
    for key in ("parser_smoke",):
        item = report.get(key, {})
        if item.get("skipped"):
            continue
        if not item.get("ok"):
            failures.append(key)

    if failures:
        print("Environment verification failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
