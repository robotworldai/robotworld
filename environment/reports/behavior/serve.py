"""Read-only report server, with MP4 ranges and no symlinks/directory listings."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_head(self):
        parts = Path(unquote(urlsplit(self.path).path).lstrip("/")).parts
        root = Path(self.directory).resolve()
        path = root
        for part in parts:
            if part in ("..", ".") or part.startswith("."):
                self.send_error(404)
                return None
            path = path / part
            if path.is_symlink():
                self.send_error(404)
                return None
        if path.is_dir():
            path = path / "index.html"
        if (path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file()
                or path.suffix.lower() not in {".html", ".css", ".js", ".json", ".png", ".mp4", ".ttf"}):
            self.send_error(404)
            return None
        size = path.stat().st_size
        start, end, status = 0, size-1, 200
        value = self.headers.get("Range")
        if value:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
            if not match or not any(match.groups()):
                self.send_error(416)
                return None
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), end) if right else end
            else:
                start = max(0, size-int(right))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            status = 206
        stream = path.open("rb")
        stream.seek(start)
        self.remaining = max(0, end-start+1)
        self.send_response(status)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(self.remaining))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; media-src 'self'; style-src 'self'; script-src 'self'; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        try:
            while self.remaining:
                chunk = source.read(min(self.remaining, 65536))
                if not chunk:
                    break
                outputfile.write(chunk)
                self.remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), partial(Handler, directory=str(args.root.resolve())))
    server.serve_forever()


if __name__ == "__main__":
    main()
