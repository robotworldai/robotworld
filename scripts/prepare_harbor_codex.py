"""Stage a verified existing Codex source build for a portable Harbor agent container."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from environment.runtime.codex_session import CodexSession

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.provider_config.read_text())
    manifest=CodexSession.verify_build(config['manifest'])
    if manifest.get('local_patch'):
        raise ValueError('Container packaging currently requires an unmodified verified Codex source build')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    source=Path(manifest['source'])
    # A shallow checkout preserves the real commit without copying local unrelated history.
    subprocess.run(['git','clone','--quiet','--depth','1',source.as_uri(),str(output/'codex')],check=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=output/'codex',text=True).strip()
    if revision!=manifest['commit']:raise ValueError('Source changed during packaging')
    remote=Path('/opt/robotworld-runtimes')
    relative=Path('var/build/codex')/revision/'debug/codex-app-server'
    target=output/relative;target.parent.mkdir(parents=True);shutil.copy2(manifest['binary'],target)
    packaged={'source':str(remote/'codex'),'commit':revision,'binary':str(remote/relative),
              'sha256':manifest['sha256']}
    (output/'build.json').write_text(json.dumps(packaged,indent=2))
    overrides=[]
    for entry in config['overrides']:
        parsed=tomllib.loads(entry)
        if 'model_catalog_json' in parsed:
            shutil.copyfile(parsed['model_catalog_json'],output/'models.json')
            overrides.append('model_catalog_json='+json.dumps(str(remote/'models.json')))
        else:overrides.append(entry)
    config={**config,'manifest':str(remote/'build.json'),'overrides':overrides}
    (output/'provider.json').write_text(json.dumps(config,indent=2))
    (output/'provenance.json').write_text(json.dumps({'commit':revision,'binary_sha256':manifest['sha256'],
       'method':'reuse verified source-built binary; shallow source clone; relocated manifest',
       'original_provider_config':str(Path(args.provider_config).resolve()),
       'original_build_manifest':str(Path(json.loads(args.provider_config.read_text())['manifest']).resolve()),
       'credential_values_included':False},indent=2))
    print(output)
if __name__=='__main__':main()
