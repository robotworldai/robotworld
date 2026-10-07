#!/usr/bin/env python3
"""Fetch the complete fixed upstream checkout, preserving all licenses/assets."""
from pathlib import Path
import json
import subprocess

ROOT=Path(__file__).resolve().parent

def main():
    lock=json.loads((ROOT/'project.json').read_text())
    checkout=ROOT/'checkout'
    if checkout.exists():
        head=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(checkout),'status','--porcelain'],text=True).strip()
        if head!=lock['commit'] or dirty:
            raise SystemExit('Existing checkout is not the clean pinned revision; refusing to overwrite it.')
    else:
        subprocess.run(['git','clone',lock['repository'],str(checkout)],check=True)
        subprocess.run(['git','-C',str(checkout),'checkout','--detach',lock['commit']],check=True)
    # LFS is used for demo/checkpoint files by upstream; keep a complete checkout.
    subprocess.run(['git','-C',str(checkout),'lfs','pull'],check=True)
    from prepare_assets import verify
    print(json.dumps({'source':str(checkout),'commit':verify()['commit']},indent=2))

if __name__=='__main__':main()
