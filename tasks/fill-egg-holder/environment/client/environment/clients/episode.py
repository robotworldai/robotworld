import json
import os
import urllib.request


def prompt_description(description):
    """Tool schemas are already delivered through native dynamic tools or MCP.

    Preserve the full describe response on disk, but do not duplicate all tool
    descriptions and schemas in both system and user messages.
    """
    if os.environ.get('WORLD_AGENT_INTERFACE', 'native-feedback-v1') != 'native-feedback-v1':
        # Generic execute does not expose each action schema as a top-level tool.
        return dict(description)
    return {**{key:value for key,value in description.items() if key != 'tools'},
            'available_robot_tools':[tool.get('name', tool.get('function', {}).get('name'))
                                     for tool in description.get('tools', [])],
            'tool_schema_reference':'Read the supplied native/MCP tool definitions for exact parameters, descriptions and limits.'}


class EpisodeClient:
    def __init__(self, url=None, token=None):
        self.url = (url or os.environ['WORLD_EPISODE_URL']).rstrip('/')
        self.token = token or os.environ['WORLD_EPISODE_TOKEN']

    def prepare_instruction(self, instruction, output):
        """Deliver the public runtime contract before the first model request.

        Uses exactly the agent-visible describe endpoint, never simulator files.
        Both harnesses archive the identical format for auditing and replay.
        """
        from pathlib import Path
        description = self.request('describe')
        if not isinstance(description, dict) or not description.get('tools'):
            raise ValueError('Episode description must expose its native action tools')
        self.description = description
        text = (instruction + '\n\n## Public runtime task and robot specification\n\n'
                'The following is the public episode specification; full action schemas are supplied separately as tool definitions. '
                'Read its task instruction, robot geometry, action schemas, units, reference frames, '
                'gains, repetition semantics and controller assistance. Query observe for current state '
                'and images before acting. If this configuration conflicts with the static task brief, '
                'report the conflict rather than guessing.\n\n```json\n' +
                json.dumps(prompt_description(description), ensure_ascii=False, indent=2, allow_nan=False) + '\n```\n')
        output = Path(output)
        (output/'initial-description.json').write_text(json.dumps(description, ensure_ascii=False, indent=2)+'\n')
        (output/'effective-instruction.md').write_text(text)
        return text

    def system_instruction(self, base, output):
        """Keep the public robot/control contract at system level for both harnesses."""
        from pathlib import Path
        description = getattr(self, 'description', None)
        if description is None:
            description = self.request('describe')
        text = base + '\n\nPublic environment control specification:\n' + json.dumps(
            prompt_description(description), ensure_ascii=False, indent=2, allow_nan=False)
        (Path(output)/'effective-system-prompt.txt').write_text(text)
        return text

    def request(self, path, body=None):
        req = urllib.request.Request(self.url+'/v1/'+path,
            data=None if body is None else json.dumps(body,allow_nan=False).encode(),
            headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
        # No automatic POST retries. Query receipt after a timeout.
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)
