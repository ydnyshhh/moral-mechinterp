from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from moral_mechinterp.cli.evaluate_behavior import app

    app()


if __name__ == "__main__":
    main()
