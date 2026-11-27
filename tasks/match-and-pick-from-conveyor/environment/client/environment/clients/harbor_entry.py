"""Task-owned startup and MCP bootstrap, independent of any agent harness."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
CONNECTION = Path('/opt/robotworld-session/connection.json')
PUBLIC_BRIEF = Path('/workspace/ROBOTWORLD.md')
KEYS = ('WORLD_EPISODE_URL', 'WORLD_EPISODE_TOKEN', 'WORLD_AGENT_INTERFACE')


def initialize(connection=CONNECTION, brief=PUBLIC_BRIEF):
    from environment.clients.episode import EpisodeClient
    client = EpisodeClient()
    description = client.request('describe')
    connection.parent.mkdir(parents=True, exist_ok=True)
    # This is only the agent's per-episode action capability, never model auth
    # or evaluator access. It is not included in logs or collected artifacts.
    connection.write_text(json.dumps({k: os.environ[k] for k in KEYS if k in os.environ}))
    connection.chmod(0o444)
    brief.parent.mkdir(parents=True, exist_ok=True)
    brief.write_text('# RobotWorld public task and robot specification\n\n'
                     'Read this specification and call observe before controlling the robot. '
                     'Only the listed action tools advance physics. Shell may process observations; '
                     'it must not bypass the action interface. Continue until status is terminal.\n\n'
                     '```json\n' + json.dumps(description, ensure_ascii=False, indent=2) + '\n```\n')
    brief.chmod(0o444)


def load_connection(path=CONNECTION):
    data = json.loads(path.read_text())
    if not data.get('WORLD_EPISODE_URL') or not data.get('WORLD_EPISODE_TOKEN'):
        raise ValueError('Task episode connection is incomplete')
    for key in KEYS:
        if key in data:
            os.environ[key] = data[key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['initialize', 'mcp'])
    args = parser.parse_args()
    if args.mode == 'initialize':
        initialize()
        os.execvp('sleep', ['sleep', 'infinity'])
    else:
        load_connection()
        import asyncio
        from environment.bridges.mcp import main as run_mcp
        asyncio.run(run_mcp())


if __name__ == '__main__':
    main()
