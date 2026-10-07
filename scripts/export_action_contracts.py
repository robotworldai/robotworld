#!/usr/bin/env python3
"""Export English model-facing control contracts without running a model/simulator.

Native tables use public metadata from existing prompt snapshots. These are
examples, not newly evaluated episodes or a substitute for runtime resolution.
"""
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from environment.benchmarks.action_contracts import action_contract
from environment.benchmarks.native_project.action_prompt import build_action_guide
from environment.benchmarks.wheeledlab.control import action_guide as car_guide


def main():
    out=ROOT/'docs/action-contracts';out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'docs/scoring/task-scoring.json').read_text())['tasks']
    entries=[]
    for bench in sorted({r['benchmark'] for r in rows}):
        tasks=[r for r in rows if r['benchmark']==bench]
        provenance=None
        if bench in ['robodojo','behavior_1k','robocasa','ai_cps']:
            text=action_contract(bench)
        elif bench=='robolab':
            text='\n\n'.join(action_contract(bench,m) for m in ['joint_position','absolute_ik','relative_ik'])
        elif bench=='wheeledlab':
            text='Illustrative 50Hz profiles. Actual prompts substitute the task control timestep.\n\n'+car_guide(.02,False)+'\n'+car_guide(.02,True)
        elif bench=='humanoid_soccer':
            text=('The runtime policy generates both system and tool contracts with action_guide(names,limits,mode,dt). '
                  'DIRECT: absolute named joint radians; repeated targets do not accumulate. Omitted ordinary-call joints latch measured qpos; omitted feedback-code joints retain previous program targets. '
                  'HYBRID: bounded radian offsets are applied to EACH new original-policy proposal, not accumulated across ticks. '
                  'Example direct q=0.1 requests 0.1rad; hybrid offset=0.1 adds 0.1rad to a fresh proposal, subject to limits. '
                  'At the nominal50Hz, 10steps=0.2s. Optional balance assistance remains disclosed. Exact joint order and limits are generated from the current model.\n')
        else:
            prompts=sorted((ROOT/'outputs'/bench).rglob('prompt.json'),key=lambda p:p.stat().st_mtime,reverse=True)
            text=None
            for p in prompts:
                prompt=json.loads(p.read_text()).get('baseInstructions','')
                if 'Native action metadata:\n' not in prompt:continue
                metadata=json.JSONDecoder().raw_decode(prompt.split('Native action metadata:\n',1)[1])[0]
                timestep=re.search(r'One control step = ([0-9.e+-]+) s',prompt)
                if not timestep:continue
                task=next((r for r in tasks if r['task'] in p.parts),tasks[0])
                identifier=task['legacy_task_id'] or task['task']
                text=build_action_guide(identifier,metadata,float(timestep[1]))
                provenance=str(p.relative_to(ROOT));break
            if text is None:
                text=('No matching public runtime metadata snapshot is available in outputs. '
                      'The active runtime still constructs its exact channel table from the instantiated action manager. '
                      'This file does not invent resolved joint order or claim a successful simulation.\n')
                if bench=='flamingo':
                    text+=('Pinned source profile: six absolute actuator position channels (scale1,offset0,rad) followed by two wheel velocity channels (scale40,offset0,rad/s). '
                           'Leg position channels are MOTOR SPACE with gear ratio -1.5, so +0.1rad motor target corresponds to a static physical target near -0.066667rad. '
                           'Other position channels use physical joint radians. Zero commands the authored zero, not the measured pose. Actuator delays remain active. '
                           'See environment/benchmarks/flamingo/project.py and native_project/action_prompt.py.\n')
        assert not re.search(r'[\u4e00-\u9fff]',text),bench
        prefix=f'# {bench}: English action contract\n\n'
        if provenance:prefix+=f'Example regenerated from public action metadata in `{provenance}`; not a new rollout.\n\n'
        (out/f'{bench}.md').write_text(prefix+text)
        entries.append({'benchmark':bench,'task_count':len(tasks),'tasks':[r['task'] for r in tasks],
                        'contract':f'{bench}.md','runtime_metadata_example_source':provenance})
    report={'benchmarks':len(entries),'tasks':len(rows),'language':'English','scope':'prompt/control semantics audit, not simulation performance', 'entries':entries}
    (out/'index.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    links='\n'.join(f"- [{e['benchmark']}]({e['contract']}) — {e['task_count']} tasks" for e in entries)
    (out/'README.md').write_text('# English action contracts\n\nGenerated with `python scripts/export_action_contracts.py`.\n\n'+links+'\n\nRuntime-generated tables remain authoritative for each rollout. Source audit: [ACTION_CONTRACT_AUDIT.md](../ACTION_CONTRACT_AUDIT.md).\n')
    print(f'Exported {len(entries)} benchmark contracts covering {len(rows)} selected tasks to {out}')


if __name__=='__main__':main()
