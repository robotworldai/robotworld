import pytest
from environment.benchmarks.robocasa.seeding import seed_config


def test_constructor_and_reset_seeds_are_distinct():
    names = [str(i) for i in range(14)] + ['CoffeeSetupMug']
    config = seed_config('CoffeeSetupMug', 7, names)
    assert config['constructor_seed'] == 7
    assert config['reset_seed'] == 21
    assert config['global_episode_index'] == 14


def test_matches_legacy_multiple_trial_schedule():
    config = seed_config('b', 7, ['a', 'b'], num_trials=50, episode_index=3)
    assert config['reset_seed'] == 60


def test_direct_and_explicit_reproduction_remain_available():
    assert seed_config('b', 7, ['a', 'b'], protocol='direct')['reset_seed'] == 7
    config = seed_config('b', 7, ['a', 'b'], reset_seed=123)
    assert config['constructor_seed'] == 7 and config['reset_seed'] == 123
    assert config['reset_seed_overridden']


@pytest.mark.parametrize('kwargs', [{'num_trials':0}, {'num_trials':True},
                                   {'episode_index':-1}, {'protocol':'unknown'}])
def test_invalid_schedules_rejected(kwargs):
    with pytest.raises(ValueError):
        seed_config('a', 7, ['a'], **kwargs)
