"""Preserve partial recordings on infrastructure abort without scoring the task."""
import json,traceback
from functools import wraps
from pathlib import Path
from environment.runtime.nonaction_budget import NonActionBudgetExceeded


def close_aborted_episode(env,output,error):
    status={'aborted':True,'reason':str(error),'official_episode_complete':False,
            'trajectory_flushed':False,'cleanup_errors':[]}
    recorder=getattr(env,'recorder_manager',None)
    try:
        if recorder is not None:
            # Flush existing samples only: export_episodes would add terminal
            # task status, which is inappropriate for an infrastructure abort.
            recorder.flush_buffer(verbose=False)
            status['trajectory_flushed']=True
    except Exception:
        status['cleanup_errors'].append(traceback.format_exc())
    if recorder is not None:
        for name in ('_dataset_file_handler','_failed_episode_dataset_file_handler'):
            handler=getattr(recorder,name,None)
            if handler is not None:
                try:
                    stream=getattr(handler,'_hdf5_file_stream',None)
                    if stream is not None:
                        stream.attrs['world_episode_aborted']=True
                        stream.attrs['world_official_episode_complete']=False
                    handler.close()
                except Exception:status['cleanup_errors'].append(traceback.format_exc())
    try:env.close()
    except Exception:status['cleanup_errors'].append(traceback.format_exc())
    Path(output).joinpath('episode-aborted.json').write_text(json.dumps(status,indent=2)+'\n')


def install_abort_cleanup(output):
    import robolab.eval.episode as episode
    original=episode.run_episode
    @wraps(original)
    def run_episode(*args,**kwargs):
        try:return original(*args,**kwargs)
        except NonActionBudgetExceeded as stop:
            env=kwargs.get('env',args[0] if args else None)
            # The upstream finally block has closed video/client. Do not invoke
            # env.step or mark this as natural native termination.
            rows=env.get_env_results()
            for row in rows:
                row['success']=row.get('success') is True
                row['step']=stop.snapshot['control_steps']
                row['interaction_budget']=stop.snapshot
                row['stop_reason']=stop.snapshot['stop_reason']
            recorder=getattr(env,'recorder_manager',None)
            if recorder is not None:recorder.flush_buffer(verbose=False)
            Path(output).joinpath('interaction-stop-result.json').write_text(json.dumps(
                {'success':any(r['success'] for r in rows),
                 'control_steps':stop.snapshot['control_steps'],
                 'stop_reason':stop.snapshot['stop_reason'],'interaction_budget':stop.snapshot},indent=2))
            return rows,[],{}
        except BaseException as error:
            close_aborted_episode(kwargs.get('env',args[0] if args else None),output,error)
            raise
    episode.run_episode=run_episode
