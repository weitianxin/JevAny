#!/usr/bin/env python3
"""Compatibility entry point; the installable builder lives in jevany.datasets."""
from jevany.datasets.build_sft import *  # noqa: F403

if __name__ == "__main__":
    main()
