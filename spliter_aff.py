#!/usr/bin/env python3
"""Arcaea 谱面（.aff）拆谱启动器：在任何目录下都能直接跑。

    spliter_aff breachoffaith_2 --stats
    spliter_aff breachoffaith_2 -o out.aff --show-constraints
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent / "py"


def main() -> int:
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from spliter.aff_cli import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
