"""Six-iteration entry. Same selector as the three-iteration runs.

The only difference is the config: six candidate columns, reference encoder
iteration 4, and the 1.7B layer index.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_main import main  # noqa: E402

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv[1:1] = ["--config", str(ROOT / "configs" / "rzero_1.7b_6iter.yaml")]
    main()
