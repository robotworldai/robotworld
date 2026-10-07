import pytest
from omegaconf import OmegaConf

from environment.evaluation.world_success.scenes import configure
from environment.evaluation.world_success.profiles import get_profile, VERSION


@pytest.mark.parametrize('bench,task,native', [
    ('omnidrones','T17',600),
    ('omnidrones','T16',500),
    ('volleybots','drone_volleyball_solo_juggle',600),
])
def test_resolved_engine_horizon_matches_approved_profile(monkeypatch,bench,task,native):
    cfg=OmegaConf.create({'task':{'env':{'max_episode_length':native}},'env':'${task.env}'})
    OmegaConf.resolve(cfg)
    monkeypatch.setenv('WORLD_SCORING_PROFILE',VERSION)
    configure(bench,task,cfg)
    expected=get_profile(bench,task)['steps']
    assert cfg.env.max_episode_length==cfg.task.env.max_episode_length==expected


def test_native_profile_horizon_unchanged(monkeypatch):
    cfg=OmegaConf.create({'task':{'env':{'max_episode_length':600}},'env':'${task.env}'})
    OmegaConf.resolve(cfg)
    monkeypatch.setenv('WORLD_SCORING_PROFILE','native')
    configure('omnidrones','T17',cfg)
    assert cfg.env.max_episode_length==cfg.task.env.max_episode_length==600
