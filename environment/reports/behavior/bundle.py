"""Bundle the reviewed report, RGB, videos and font into one offline HTML."""
import argparse
import base64
import json
from pathlib import Path


MIME = {".png": "image/png", ".mp4": "video/mp4", ".json": "application/json", ".ttf": "font/ttf"}


def script_json(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def bundle(root, destination):
    root = root.resolve()
    data = json.loads((root / "data.json").read_text())
    assets = {}

    def add(relative):
        path = root / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(root)
                or path.suffix not in MIME or not path.is_file()):
            raise ValueError("Refuse asset outside reviewed report")
        if relative not in assets:
            assets[relative] = {"type": MIME[path.suffix],
                                "base64": base64.b64encode(path.read_bytes()).decode("ascii")}

    for task in data["tasks"]:
        for url in task["videos"].values():
            add(url)
        for observation in task["observations"]:
            for url in observation["images"].values():
                add(url)
        add(task["download"])
    font = root / "fonts/NotoSansSC.ttf"
    css = (root / "style.css").read_text()
    if font.is_file():
        add("fonts/NotoSansSC.ttf")
        css = css.replace('url("fonts/NotoSansSC.ttf")',
                          'url("data:font/ttf;base64,' + assets.pop("fonts/NotoSansSC.ttf")["base64"] + '")')
    else:
        raise ValueError("Missing local font; cannot build self-contained report")
    bootstrap = """
(() => {
  const assetNode = document.getElementById('report-assets');
  const assets = JSON.parse(assetNode.textContent), urls = new Map();
  function assetUrl(path) {
    if (!urls.has(path)) {
      const asset = assets[path];
      if (!asset) throw new Error('Missing embedded asset: ' + path);
      const bytes = Uint8Array.from(atob(asset.base64), c => c.charCodeAt(0));
      urls.set(path, URL.createObjectURL(new Blob([bytes], {type: asset.type})));
    }
    return urls.get(path);
  }
  const reportNode = document.getElementById('report-data');
  const report = JSON.parse(reportNode.textContent);
  for (const task of report.tasks) {
    for (const key of Object.keys(task.videos)) task.videos[key] = assetUrl(task.videos[key]);
    for (const observation of task.observations) {
      for (const key of Object.keys(observation.images)) observation.images[key] = assetUrl(observation.images[key]);
    }
    task.download = assetUrl(task.download);
  }
  reportNode.textContent = JSON.stringify(report);
  assetNode.remove();
  addEventListener('pagehide', event => { if (!event.persisted) for (const url of urls.values()) URL.revokeObjectURL(url); });
})();
"""
    html = (root / "index.html").read_text()
    css_tag = '<link rel="stylesheet" href="style.css">'
    js_tag = '<script src="app.js" defer></script>'
    if html.count(css_tag) != 1 or html.count(js_tag) != 1:
        raise ValueError("Unexpected report entrypoint")
    html = html.replace(css_tag, '<style>' + css + '</style>').replace(js_tag, '')
    policy = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
              "img-src blob: data:; media-src blob: data:; font-src data:; "
              "connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'")
    html = html.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n'
                        '<meta http-equiv="Content-Security-Policy" content="' + policy + '">')
    html = html.replace('RUN REVIEW</a>', 'OFFLINE REVIEW</a>')
    app = (root / "app.js").read_text()
    if '</script' in app.lower() or '</style' in css.lower():
        raise ValueError("Unsafe inline script/style terminator")
    scripts = ('<script id="report-data" type="application/json">' + script_json(data) + '</script>\n'
               '<script id="report-assets" type="application/json">' + script_json(assets) + '</script>\n'
               '<script>' + bootstrap + '</script>\n<script>' + app + '</script>\n')
    html = html.replace('</body>', scripts + '</body>')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.tmp')
    temp.write_text(html)
    temp.replace(destination)
    return {"file": str(destination), "bytes": destination.stat().st_size,
            "embedded_assets": len(assets) + 1, "tasks": len(data["tasks"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bundle(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
