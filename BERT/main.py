import argparse
import json
import os
import sys

from pipeline.analyzer import analyze_paper

SUPPORTED_EXTS = {".pdf", ".docx"}


def main():
    parser = argparse.ArgumentParser(
        description="GAD Thesis — Academic Paper Analyzer (CLI)"
    )
    parser.add_argument("file", help="Path to a .pdf or .docx file")
    parser.add_argument(
        "--threshold", type=float, default=0.7,
        help="Classification threshold (default: 0.7)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print result as JSON instead of human-readable summary",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found — {args.file}", file=sys.stderr)
        sys.exit(1)

    ext = os.path.splitext(args.file)[1].lower()
    if ext not in SUPPORTED_EXTS:
        print(
            f"Error: Unsupported file type '{ext}'. Use .pdf or .docx",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs("output", exist_ok=True)
    result = analyze_paper(args.file, threshold=args.threshold)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_summary(result)


def _print_summary(result: dict) -> None:
    print("\n" + "=" * 75)
    print(f"OVERALL SCORE: {result['overallLabel']}")
    print("=" * 75)
    stats = result["stats"]
    print(f"Total sentences   : {stats['totalSentences']}")
    print(f"Flagged sentences : {stats['flaggedSentences']}")
    print(
        f"Authors detected  : {stats['maleNames']} male, "
        f"{stats['femaleNames']} female, {stats['unknownNames']} unknown"
    )
    print(f"\nFlagged rows ({len(result['rows'])}):")
    for row in result["rows"][:20]:
        phrase = row["phrase"]
        if len(phrase) > 60:
            phrase = phrase[:57] + "..."
        print(f"  [{row['aspect']:<18}] {row['score']:>6}  {phrase}")
    if len(result["rows"]) > 20:
        print(f"  ... and {len(result['rows']) - 20} more")


if __name__ == "__main__":
    main()
