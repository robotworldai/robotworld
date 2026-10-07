import json
from types import SimpleNamespace
from environment.benchmarks.robolab.lifecycle import close_aborted_episode

def test_abort_flushes_raw_samples_without_exporting_task_outcome(tmp_path):
    calls=[];attrs={}
    class Recorder:
        _dataset_file_handler=SimpleNamespace(close=lambda:calls.append('close_hdf5'),_hdf5_file_stream=SimpleNamespace(attrs=attrs))
        _failed_episode_dataset_file_handler=None
        def flush_buffer(self,*,verbose):calls.append('flush_raw')
        def export_episodes(self):raise AssertionError('Must not fabricate terminal task status')
    env=SimpleNamespace(recorder_manager=Recorder(),close=lambda:calls.append('close_env'))
    close_aborted_episode(env,tmp_path,TimeoutError('tool budget'))
    assert calls==['flush_raw','close_hdf5','close_env']
    status=json.loads((tmp_path/'episode-aborted.json').read_text())
    assert status['trajectory_flushed'] and not status['official_episode_complete']
    assert not status['cleanup_errors']
    assert attrs=={'world_episode_aborted':True,'world_official_episode_complete':False}
