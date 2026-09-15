#!/usr/bin/env python3
"""
Marigold-360 Panorama Inference CLI Entrypoint
Cross-platform startup script for local environments and Docker containers.
"""
import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
_root_dir = str(Path(__file__).resolve().parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from standalone_marigold.infer_panorama import main

if __name__ == "__main__":
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        sys.argv.append("--help")

    # Call Click CLI main
    main()
