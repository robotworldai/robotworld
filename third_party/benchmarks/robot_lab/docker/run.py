"""Source-built local Codex + native robot_lab in Docker."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from environment.runtime.native_project_launch import main
if __name__=='__main__':main('robot_lab')
