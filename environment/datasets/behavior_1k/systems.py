"""Resolve particle dependencies declared by the selected objects' BDDL abilities."""
import json
from pathlib import Path
from .subset import fetch


def prepare_registry(index, destination):
    """Register global recipe names while keeping unrelated model bodies absent."""
    files = json.loads(Path(index).read_text())['files']
    selected = [f['path'] for f in files if f['path'].startswith('systems/')
                and f['path'].count('/') == 2 and f['path'].endswith('/metadata.json')]
    report = fetch(index, destination / 'behavior-1k-assets', selected)
    (destination / 'system-registry-manifest.json').write_text(json.dumps(report, indent=2))
    models = {}
    for f in files:
        parts = f['path'].split('/')
        if len(parts) > 3 and parts[0] == 'systems':
            models.setdefault(parts[1], set()).add(parts[2])
    chosen = {name: sorted(choices)[0] for name, choices in models.items()}
    for name, model in chosen.items():
        # Preserve original sorted model selection. Actual unprepared use fails
        # with FileNotFoundError rather than silently using generic particles.
        (destination / 'behavior-1k-assets/systems' / name / model).mkdir(parents=True, exist_ok=True)
    (destination / 'system-model-index.json').write_text(json.dumps(chosen, indent=2))


def select_system_files(categories, taxonomy_path, files):
    nodes = {}

    def walk(value):
        if isinstance(value, dict):
            if 'name' in value and 'abilities' in value:
                nodes[value['name']] = value
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(json.loads(taxonomy_path.read_text()))
    references = set()

    def strings(value):
        if isinstance(value, str):
            references.add(value)
        elif isinstance(value, dict):
            for key, child in value.items():
                references.add(key)
                strings(child)
        elif isinstance(value, list):
            for child in value:
                strings(child)

    for node in nodes.values():
        if categories.intersection(node.get('categories', [])):
            strings({k: v for k, v in node['abilities'].items() if 'particle' in k.lower()})
    systems = {s for ref in references if ref in nodes for s in nodes[ref].get('substances', [])}
    selected = []
    for name in sorted(systems):
        prefix = f'systems/{name}/'
        models = sorted({f['path'][len(prefix):].split('/')[0] for f in files
                         if f['path'].startswith(prefix) and '/' in f['path'][len(prefix):]})
        # Upstream create_system_from_metadata chooses sorted(particle_assets)[0].
        selected.extend(f['path'] for f in files if f['path'] == prefix + 'metadata.json'
                        or (models and f['path'].startswith(prefix + models[0] + '/')))
    return sorted(systems), selected
