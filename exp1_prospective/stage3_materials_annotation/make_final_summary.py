#!/usr/bin/env python3
"""Create the canonical closed Stage 3 descriptive summary."""

from __future__ import annotations

import sys

if __package__:
    from . import make_interim_snapshot
else:
    import make_interim_snapshot


def main() -> int:
    if "--final-descriptive" not in sys.argv:
        sys.argv.append("--final-descriptive")
    return make_interim_snapshot.main()


if __name__ == "__main__":
    raise SystemExit(main())
