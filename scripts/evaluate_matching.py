"""Script to evaluate semantic matching quality.

Usage:
    python scripts/evaluate_matching.py

Implementation planned for Milestone 3.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    """Evaluate matching accuracy against labeled examples."""
    print("Matching evaluation will be implemented in Milestone 3.")
    print("This script will:")
    print("  1. Load hand-labeled resume/JD pairs")
    print("  2. Run the semantic matcher")
    print("  3. Compare predicted matches against labels")
    print("  4. Report precision, recall, and F1 score")


if __name__ == "__main__":
    main()
