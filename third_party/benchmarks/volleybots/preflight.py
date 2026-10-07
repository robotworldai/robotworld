"""Read-only source/asset readiness check, no model or GPU session."""
from pathlib import Path
import sys,json,subprocess,importlib.util
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[2]))
config=json.loads((ROOT/'project.json').read_text())
checks={}
try:
    assert subprocess.check_output(['git','-C',str(ROOT/'checkout'),'rev-parse','HEAD'],text=True).strip()==config['commit']
    assert not subprocess.check_output(['git','-C',str(ROOT/'checkout'),'status','--porcelain'],text=True).strip()
    checks['source']='complete pinned clean checkout'
    if config['key']=='omniisaacgymenvs':
        from environment.benchmarks.omniisaacgymenvs.project import preflight
        preflight()
    else:
        spec=importlib.util.spec_from_file_location('asset_check',ROOT/'prepare_assets.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main()
    checks['assets']='local bytes verified; not a GPU render check'
    if config['key']=='volleybots':
        from environment.benchmarks.volleybots.project import preflight
        preflight()
        checks['opponent']='hash verified'
except Exception as e:
    checks['blocked']=str(e)
checks['runtime']='not executed by this preflight'
print(json.dumps(checks,indent=2,ensure_ascii=False))
raise SystemExit(1 if 'blocked' in checks else 0)
