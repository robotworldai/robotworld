"""Append-only, flushed episode evidence; image payloads remain exact in RPC records."""
import json
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path


def without_images(value):
    """Keep event structure and image positions, replacing inline image bytes."""
    if isinstance(value, str) and value.startswith('data:image/'):
        mime = value.split(';', 1)[0].split(',', 1)[0][5:]
        digest = hashlib.sha256(value.encode()).hexdigest()
        return f'[image omitted: mime={mime}; data_url_sha256={digest}; chars={len(value)}]'
    if isinstance(value, str) and '"image_pixels"' in value:
        # Tool reports may contain returned camera memory as JSON text.
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            pass
        else:
            if isinstance(parsed, (dict, list)):
                return json.dumps(without_images(parsed), ensure_ascii=False)
    if isinstance(value, dict):
        if value.get('type') == 'image_pixels' and isinstance(value.get('pixels'), list):
            raw = json.dumps(value['pixels'], separators=(',', ':')).encode()
            return {**{key: without_images(item) for key, item in value.items() if key != 'pixels'},
                    'pixels': f'[image pixels omitted: sha256={hashlib.sha256(raw).hexdigest()}; values={len(value["pixels"])}]'}
        return {key: without_images(item) for key, item in value.items()}
    if isinstance(value, list):
        return [without_images(item) for item in value]
    return value


def export_without_images(directory):
    """Export completed logs without modifying originals; atomic per-file replacement."""
    directory = Path(directory)
    output = directory / 'no-images'
    output.mkdir(parents=True, exist_ok=True)
    for source in sorted(directory.glob('*.jsonl')):
        target = output / source.name
        temporary = target.with_suffix('.jsonl.tmp')
        try:
            with source.open(encoding='utf-8') as reader, temporary.open('w', encoding='utf-8') as writer:
                for line in reader:
                    writer.write(json.dumps(without_images(json.loads(line)), ensure_ascii=False) + '\n')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.text_path = self.path.parent / 'no-images' / self.path.name
        self.text_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.sequence = 0

    def write(self, kind, payload):
        def encode(value):
            if hasattr(value, 'tolist'):
                return value.tolist()
            raise TypeError(type(value).__name__)
        with self.lock:
            record = {'sequence': self.sequence, 'timestamp': datetime.now(timezone.utc).isoformat(),
                      'kind': kind, 'payload': payload}
            for path, data in ((self.path, record), (self.text_path, without_images(record))):
                with path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(data, default=encode, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
            self.sequence += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export completed event logs without inline images.')
    parser.add_argument('events_directory', type=Path)
    export_without_images(parser.parse_args().events_directory)
