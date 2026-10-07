import runpy
from pathlib import Path

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/prepare_docker.py'))


def test_all_selected_runtime_profiles_have_recipes():
    recipes, selected = MODULE['recipes']()
    assert len(selected) == 20
    tags = {tag for group in selected.values() for tag in group}
    assert len(tags) == 21
    assert tags <= recipes.keys()
    assert set(selected['volleybots']) == {
        'world/volleybots:isaac2023.1.0-hotfix1',
        'world/volleybots:isaac6.0.1-experimental',
    }


def test_arg_and_multistage_local_dependencies():
    parse = MODULE['local_dependencies']
    assert parse('ARG BASE_IMAGE=world/base:v1\nFROM ${BASE_IMAGE}\nFROM world/other:v2 AS build') == [
        'world/base:v1', 'world/other:v2']
    assert parse('FROM python:3.11\nFROM world/base:v1') == ['world/base:v1']


def test_unresolved_base_is_not_silently_skipped():
    import pytest
    with pytest.raises(ValueError, match='Unresolved'):
        MODULE['local_dependencies']('FROM ${MISSING}')
