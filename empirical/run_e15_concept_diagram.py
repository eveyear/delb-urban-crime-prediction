from __future__ import annotations

import sys
from pathlib import Path


EMPIRICAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EMPIRICAL_ROOT / "src"))

from entropy_crime_bike.e15_concept_diagram import main  # noqa: E402


if __name__ == "__main__":
    main()
