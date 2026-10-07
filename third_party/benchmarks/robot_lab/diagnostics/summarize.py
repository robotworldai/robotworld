"""Summarize completed diagnostic cases without treating them as task scores."""
import argparse
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cases={}
    for root in a.roots:
        index={r['case']:r for r in json.loads((root/'index.json').read_text())} if (root/'index.json').exists() else {}
        for directory in sorted(root.iterdir()):
            result=directory/'result.json'
            if result.exists() and not (directory/'error.txt').exists():
                launch=index.get(directory.name,{})
                cases[directory.name]={'path':str(directory),'summary':json.loads(result.read_text()),
                    'launcher_returncode':launch.get('returncode'),
                    'clean_process_exit':launch.get('returncode')==0 if 'returncode' in launch else None}
    def rows(name):
        return [json.loads(line) for line in (Path(cases[name]['path'])/'physics.jsonl').read_text().splitlines()]
    def params(name): return json.loads((Path(cases[name]['path'])/'effective-parameters.json').read_text())
    comparisons={}
    pairs=[('native-replay-a','native-replay-b'),('fixed-hold-original','fixed-hold-compatible'),
           ('fixed-pulse-original','fixed-pulse-compatible'),('fixed-replay-original','fixed-replay-compatible')]
    for control in ['hold','pulse','replay']:
        pairs += [(f'isaac45-cpu-{control}',f'isaac6-cpu-{control}'),
                  (f'isaac6-cpu-{control}',f'fixed-{control}-compatible'),
                  (f'isaac45-lab22-cpu-{control}',f'isaac6-cpu-{control}'),
                  (f'isaac45-lab22-cpu-{control}',f'isaac45-cpu-{control}'),
                  (f'isaac45-lab22-cpu-{control}',f'isaac6-oldusd-cpu-{control}'),
                  (f'isaac6-oldusd-cpu-{control}',f'isaac6-cpu-{control}')]
    for left,right in pairs:
        if left not in cases or right not in cases: continue
        l,r=rows(left),rows(right);lp,rp=params(left),params(right)
        li={n:i for i,n in enumerate(lp['joint_names'])};ri={n:i for i,n in enumerate(rp['joint_names'])}
        assert li.keys()==ri.keys(), 'Joint sets differ'
        assert [x['step'] for x in l]==[x['step'] for x in r], 'Step indices differ'
        assert [x['action'] for x in l]==[x['action'] for x in r], 'Action streams differ'
        errors=[abs(x['joint_pos'][li[n]]-y['joint_pos'][ri[n]]) for x,y in zip(l,r) for n in li]
        gravity=[math.dist(x['gravity'],y['gravity']) for x,y in zip(l,r)]
        root=[math.dist(x['root_state'][:3],y['root_state'][:3]) for x,y in zip(l,r)]
        comparisons[left+' vs '+right]={'compared_states':min(len(l),len(r)),
            'max_joint_angle_difference_rad':max(errors),'rms_joint_angle_difference_rad':math.sqrt(sum(x*x for x in errors)/len(errors)),
            'max_root_position_difference_m':max(root),'max_gravity_vector_difference':max(gravity),
            'actual_physx_drives_left':{k:lp['physx'].get(k) for k in ['get_dof_stiffnesses','get_dof_dampings']},
            'actual_physx_drives_right':{k:rp['physx'].get(k) for k in ['get_dof_stiffnesses','get_dof_dampings']}}
    result={'cases':cases,'paired_comparisons':comparisons,
            'interpretation_limit':'Diagnostics, not benchmark scores. Cases use the device, asset and framework specified by command.json. Even shared-USD CPU comparisons retain runtime/library differences and do not establish official GPU equivalence. Original merged calf contact includes foot contact and is not a valid nonfoot failure label. A completed physics trace does not imply clean process exit; check launcher_returncode.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{m:v for m,v in values.items() if not m.startswith('actual_')} for k,values in comparisons.items()},indent=2))


if __name__=='__main__':main()
