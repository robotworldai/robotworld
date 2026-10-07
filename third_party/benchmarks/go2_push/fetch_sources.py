"""Fetch the two unchanged pinned upstream repositories for T10."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parent

def main():
    lock=json.loads((ROOT/'project.json').read_text())
    rows=[(ROOT/'checkout',lock['repository'],lock['commit']),
          (ROOT/'isaaclab211','https://github.com/isaac-sim/IsaacLab.git','90b79bb2d44feb8d833f260f2bf37da3487180ba')]
    for path,repo,commit in rows:
        if not path.exists():
            subprocess.run(['git','clone',repo,str(path)],check=True)
            subprocess.run(['git','-C',str(path),'checkout','--detach',commit],check=True)
        from prepare_assets import verify_checkout
        verify_checkout(path,commit)
    print('Both pinned source trees verified')
if __name__=='__main__':main()
