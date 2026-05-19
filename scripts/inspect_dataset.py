from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# pyrefly: ignore [missing-import]
from utils import load_yaml, repo_root, resolve_from_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the Mayo dataset structure.")
    parser.add_argument(
        "--paths",
        default="configs/paths.yaml",
        help="Path to the YAML file containing dataset paths.",
    )
    return parser.parse_args()


def count_pngs_by_patient(split_dir: Path) -> OrderedDict[str, int]:
    counts: OrderedDict[str, int] = OrderedDict()
    for patient_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        counts[patient_dir.name] = len(sorted(patient_dir.glob("*.png")))
    return counts


def main() -> None:
    args = parse_args()
    paths_file = resolve_from_root(args.paths)
    paths = load_yaml(paths_file)
    train_dir = resolve_from_root(paths["train_dir"])
    test_dir = resolve_from_root(paths["test_dir"])

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    train_counts = count_pngs_by_patient(train_dir)
    test_counts = count_pngs_by_patient(test_dir)

    print("Mayo dataset inspection")
    print(f"Paths config:    {paths_file}")
    print(f"Train directory: {train_dir}")
    print(f"Test directory:  {test_dir}")
    print()
    print("Train patients:")
    for patient, count in train_counts.items():
        print(f"  {patient}: {count} png")
    print()
    print("Test patients:")
    for patient, count in test_counts.items():
        print(f"  {patient}: {count} png")
    print()
    print(f"Total train images: {sum(train_counts.values())}")
    print(f"Total test images:  {sum(test_counts.values())}")
    print(f"Total images:       {sum(train_counts.values()) + sum(test_counts.values())}")


if __name__ == "__main__":
    main()
