"""Run the upstream RTAMT monitor in its ANTLR-isolated interpreter."""
import json, math, subprocess
from pathlib import Path

def peg_insertion_check(trace, complete):
    """Human-approved XY and independent Z bounds over samples 250:299.

    This is a geometric check, not contact/mesh penetration certification.
    Native observations contain tool xyz at 18:21 and tool-minus-hole at 25:28.
    Retain XY <= .1 m; require .38 <= reference z <= .40 m.
    No additional orientation threshold is imposed.
    """
    samples = [trace[1], *trace[1:]] if len(trace) > 1 else trace
    window = samples[249:299]
    valid = complete and len(window) == 50
    distances = []
    heights = []
    for row in window:
        if len(row) < 28 or not all(math.isfinite(v) for v in (row[20], row[25], row[26])):
            valid = False
            continue
        distances.append(math.hypot(row[25], row[26]))
        heights.append(row[20])
    return {'version': 'peg-xy-z-window-v2', 'valid': valid,
            'success': all(d <= .1 and .38 <= z <= .40 for d, z in zip(distances, heights)) if valid else None,
            'samples': len(distances), 'window': [250, 299],
            'z_range_m': [.38, .40], 'xy_distance_threshold_m': .1,
            'max_xy_distance_m': max(distances) if distances else None,
            'min_z_m': min(heights) if heights else None,
            'max_z_m': max(heights) if heights else None,
            'orientation_threshold': None,
            'definition': 'Human-approved: XY distance <= .1 m and reference z in [.38,.40] m throughout trace indices 250:299; no added orientation threshold; geometric, not contact certification.'}

def score(task_name,trace,complete,source,python='/opt/rtamt/bin/python',*,verify_insertion=False):
    insertion = peg_insertion_check(trace, complete) if task_name=='FrankaPegInHole' and verify_insertion else None
    offset=25 if task_name=='FrankaPegInHole' else 21
    if complete:
        # Upstream compute_trace aliases trace=states, then states.copy_(next_states)
        # before its first vstack: slot0 becomes the first post-action observation.
        trace=[trace[1],*trace[1:]] if len(trace)>1 else trace
        import torch
        values=torch.linalg.vector_norm(torch.tensor(trace,dtype=torch.float32)[:,offset:offset+2],dim=-1).tolist()
    else:values=[math.hypot(row[offset],row[offset+1]) for row in trace]
    indexed=[[i+1,value] for i,value in enumerate(values)]
    distances=[x[1] for x in indexed]
    result={'trace_samples':len(trace),'native_episode_complete':complete,'distance':'XY metres (upstream predicate, not physical insertion/capture certification)',
      'dangerous_rate':sum(d>(.37 if task_name=='FrankaPegInHole' else .2) for d in distances)/len(distances)}
    if not complete:
        return {**result,'success':None,'robustness':None,'status':'incomplete_diagnostic'}
    if task_name=='FrankaPegInHole':completion=next((i for i,d in enumerate(distances) if d<=.1),None)
    else:completion=next((i for i in range(len(distances)-5) if all(d<=.1 for d in distances[i:i+5])),None)
    result.update(completion_time=completion,completion_time_units='zero-based trace index, not seconds; null represents upstream mean(empty)=NaN',trace_assembly='upstream first-state alias preserved; native_trace.json retains raw reset observation')
    worker=Path(__file__).with_name('score_worker.py')
    payload={'task':task_name,'source':str(source),'trace':indexed}
    reply=subprocess.run([python,str(worker)],input=json.dumps(payload),text=True,capture_output=True,timeout=60,check=True)
    result = {**result, **json.loads(reply.stdout)}
    if insertion is not None:
        result.update(native_success=result['success'], insertion_evaluation=insertion,
                      success=insertion['success'], scoring_profile=insertion['version'],
                      success_definition=insertion['definition'])
    return result
