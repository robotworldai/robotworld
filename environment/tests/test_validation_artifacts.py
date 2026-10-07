import json
from environment.validation.campaign import verify_artifacts


def test_zero_process_exit_does_not_hide_native_startup_error(tmp_path):
    (tmp_path/'exit.json').write_text(json.dumps({'returncode':0,'infrastructure_ok':False,'error':'missing sensor'}))
    (tmp_path/'result.json').write_text('{"success":false}')
    assert verify_artifacts('flamingo','T15',tmp_path)==(False,'missing sensor')


def test_unsuccessful_task_with_complete_result_is_a_valid_run(tmp_path):
    (tmp_path/'exit.json').write_text('{"returncode":0}')
    (tmp_path/'result.json').write_text('{"success":false,"terminated":true}')
    assert verify_artifacts('reflexbench','T03',tmp_path)==(True,None)


def test_missing_result_is_not_completion(tmp_path):
    (tmp_path/'exit.json').write_text('{"returncode":0}')
    assert not verify_artifacts('bench2dex','41',tmp_path)[0]


def test_failure_artifact_overrides_earlier_partial_result(tmp_path):
    (tmp_path/'result.json').write_text('{"success":false}')
    (tmp_path/'failure.json').write_text('{"error":"camera failed"}')
    assert not verify_artifacts('robodojo','hang_mugs',tmp_path)[0]
