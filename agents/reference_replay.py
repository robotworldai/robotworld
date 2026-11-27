"""Diagnostic only: replay saved public tool requests through the live MCP bridge."""
import argparse
import asyncio
import json
from pathlib import Path
from environment.clients.episode import EpisodeClient
from environment.bridges.mcp import call, tools


def reference_step_count(result):
    value=result.get('control_steps')
    if type(value) is int:
        return value, 'result.control_steps'
    value=(result.get('interaction_budget') or {}).get('control_steps')
    if type(value) is int:
        return value, 'result.interaction_budget.control_steps'
    return None, 'unavailable'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    client=EpisodeClient();description=client.request('describe')
    reference=next((r for r in json.loads(args.references.read_text())
        if r['benchmark']==description['benchmark'] and str(description['task']) in r['task_aliases']),None)
    if reference is None:raise ValueError('No historical reference for live task')
    names={t.name for t in asyncio.run(tools())}
    first=client.request('observe');first['observation'].pop('images',None)
    (args.output/'initial.json').write_text(json.dumps(first,indent=2))
    (args.output/'description.json').write_text(json.dumps(description,indent=2))
    errors=[]; expected_rejections=[]
    with (args.output/'replay.jsonl').open('w',buffering=1) as log:
        for i,command in enumerate(reference.get('command_batches', reference['commands'])):
            if client.request('status')['terminal']:break
            if 'command_batches' in reference:
                import uuid
                state=client.request('status')
                receipt=client.request('execute',dict(request_id='replay-'+uuid.uuid4().hex,
                    episode_id=state['episode_id'],observation_id=state['observation_id'],commands=command))
                log.write(json.dumps({'index':i,'commands':command,'receipt':receipt})+'\n')
                if receipt.get('error') or receipt.get('execution_uncertain'):
                    errors.append({'index':i,'error':'Replay execution error or uncertainty'})
                    break
                continue
            name=command['tool']
            if name not in names:
                errors.append({'index':i,'error':'tool_not_exposed','tool':name});break
            try:
                blocks=asyncio.run(call(name,command['arguments']))
                texts=[b.text for b in blocks if b.type=='text']
                log.write(json.dumps({'index':i,'command':command,'response_text':texts,
                                      'images':sum(b.type=='image' for b in blocks)})+'\n')
            except ValueError as exc:
                if command.get('expected_rejection'):
                    expected_rejections.append({'index':i,'error':str(exc)})
                    log.write(json.dumps({'index':i,'expected_rejection':str(exc)})+'\n')
                    continue
                errors.append({'index':i,'error':str(exc)});break
            except Exception as exc:
                # Stop on uncertainty. Never skip an uncertain action or retry it.
                errors.append({'index':i,'error':str(exc)});break
    final=client.request('status')
    expected_steps, step_source=reference_step_count(reference['expected'])
    (args.output/'summary.json').write_text(json.dumps({
        'classification':'historical_reference_replay_not_model_evaluation',
        'historical_id':reference['historical_id'],'historical_result':reference['expected'],
        'environment':final,'errors':errors,'expected_rejections':expected_rejections,
        'same_success':final['success']==reference['expected']['success'],
        'same_control_steps':None if expected_steps is None else final['control_step']==expected_steps,
        'expected_control_steps':expected_steps,'expected_step_source':step_source,
        'scope':'Public native tools via MCP, HTTP, Episode executor and configured environment checker. No model calls or state injection.'},indent=2))

if __name__=='__main__':main()
