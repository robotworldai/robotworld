#!/usr/bin/env python3
"""Export supported RobotWorld tasks as standard Harbor task directories."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environment.harbor_tasks.export import main

if __name__ == '__main__':
    main()
