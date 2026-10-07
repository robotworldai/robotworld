#!/usr/bin/env python3
"""Local source Codex + native Wheel-Legged-Lab Docker episode."""
from pathlib import Path
import sys

WORLD = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORLD))
from environment.runtime.native_project_launch import main

if __name__ == '__main__':
    main('wheel_legged')
