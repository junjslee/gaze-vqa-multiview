#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point for MVGT → Gaze-VQA benchmark builder.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaze_vqa.pipeline.main import main


if __name__ == "__main__":
    main()
