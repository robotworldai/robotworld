#!/usr/bin/env python3
"""Configure an explicit Responses-compatible model provider without saving secrets."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environment.runtime.model_config import toml_text


def configure(destination, base_url, model, key_env='WORLD_MODEL_API_KEY', reasoning=None):
    url = urlsplit(base_url)
    if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Use an HTTP(S) API base URL without credentials, query parameters, or fragments')
    if not model.strip() or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key_env):
        raise ValueError('A model ID and a valid API-key environment variable name are required')
    config = {'model': model, 'model_provider': 'api', 'model_providers': {'api': {
        'name': 'Configured model API', 'base_url': base_url.rstrip('/'),
        'wire_api': 'responses', 'env_key': key_env,
        'requires_openai_auth': False, 'supports_websockets': False}}}
    if reasoning:
        config['model_reasoning_effort'] = reasoning
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination.chmod(0o700)
    target = destination / 'config.toml'
    # Refuse to overwrite another provider or account configuration.
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(toml_text(config))
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True, help='API prefix, e.g. https://api.example.org/v1')
    p.add_argument('--model', required=True)
    p.add_argument('--key-env', default='WORLD_MODEL_API_KEY')
    p.add_argument('--output', type=Path, default=Path('var/auth/api'))
    p.add_argument('--reasoning-effort', choices=['minimal', 'low', 'medium', 'high', 'xhigh'])
    a = p.parse_args()
    try:
        path = configure(a.output, a.base_url, a.model, a.key_env, a.reasoning_effort)
    except (ValueError, FileExistsError) as e:
        p.error(str(e))
    print(f'Configuration: {path.resolve()}')
    print(f'Set {a.key_env} in your environment; its value is not stored in this file.')
    print('Set CODEX_AUTH_HOME to this directory, then run scripts/check_api.py.')


if __name__ == '__main__':
    main()
