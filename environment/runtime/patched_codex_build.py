"""Explicit fingerprint for the local image-window Codex variant; never the stock manifest."""
import hashlib
import json
from pathlib import Path
import subprocess

PATCH_FILES = {
    'codex-rs/core/src/context_manager/history.rs',
    'codex-rs/core/src/context_manager/robot_image_window.rs',
    'codex-rs/core/src/context_manager/robot_image_window_tests.rs',
}


def source_fingerprint(source):
    changed = subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=source,text=True).splitlines()
    untracked = subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=source,text=True).splitlines()
    if set(changed+untracked) != PATCH_FILES:
        raise ValueError('Unexpected changes in local Codex variant')
    payload = {name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in sorted(PATCH_FILES)}
    diff = subprocess.check_output(['git','diff','HEAD','--binary'],cwd=source)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()+diff).hexdigest()


def main():
    world = Path(__file__).resolve().parents[2]
    source = world/'codex'
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    target = world/'var/build/codex'/(commit+'-image-window-work')
    binary = target/'debug/codex-app-server-image-window-v1'
    record = dict(source=str(source),commit=commit,binary=str(binary),
                  sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  local_patch=dict(id='robot-image-window-v1',sha256=source_fingerprint(source)),
                  locked=True)
    path = target/'image-window-build-v1.json'
    if path.exists():
        raise FileExistsError('Refusing to overwrite a variant build record')
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
