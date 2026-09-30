"""Check saved paper results and, optionally, the installed reader."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def check_saved_results() -> None:
    from src.reproduction.config import SOURCE_PAPER_DIR

    from src.reproduction.materials import check_materials

    check_materials()
    result = subprocess.run(
        [sys.executable, "-B", "-m", "src.reproduction.check_numbers",
         "--paper-dir", str(SOURCE_PAPER_DIR)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(result.stdout.strip() or "Paper verification failed")
    print("PASS: paper materials and saved results")


def check_parser() -> None:
    from src.pipeline import init_default_pipeline

    pipeline = init_default_pipeline(use_gpu=False, debug_errors=True)
    cases = [
        ("PROTEIN0 binds PROTEIN1.", {(0, 1)}),
        ("PROTEIN0 is present. PROTEIN1 is present.", set()),
    ]
    for sentence, expected in cases:
        record = pipeline.process_sentence(sentence)
        if record is None:
            raise RuntimeError(f"Reader returned no result: {sentence}")
        actual = set(record.predicted_pairs)
        if actual != expected:
            raise RuntimeError(f"{sentence}: expected {sorted(expected)}, got {sorted(actual)}")
        print(f"PASS: {sentence} -> {sorted(actual)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parser", action="store_true", help="Also load Stanza, coreference and QANom on CPU")
    args = parser.parse_args()
    try:
        check_saved_results()
        if args.parser:
            check_parser()
    except RuntimeError as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
