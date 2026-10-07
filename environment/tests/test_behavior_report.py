from functools import partial
from http.server import ThreadingHTTPServer
import json
import threading
import urllib.error
import urllib.request

import pytest

from environment.reports.behavior.export import export_task, public_text
from environment.reports.behavior.serve import Handler
from environment.reports.behavior.bundle import bundle, script_json


def test_report_omits_private_config_and_only_copies_selected_files(tmp_path):
    run, output = tmp_path / "run", tmp_path / "public"
    run.mkdir()
    (run / "result.json").write_text(json.dumps(dict(
        status="completed", score_valid=True, model_score=0, source_hashes={"/root/secret": "hidden"},
        api_key="do-not-publish", selection=dict(scene="test", private_path="secret"))))
    (run / "auth.json").write_text("secret")
    (run / "request-audit.json").write_text(json.dumps([dict(started_at=1, finished_at=2,
        terminal_event="response.completed", violations=[], authorization="private")]))
    task = export_task(run, output, 1)
    text = json.dumps(task)
    assert all(word not in text for word in ("do-not-publish", "private_path", "authorization", "/root/secret"))
    assert not (output / "auth.json").exists()
    assert task["api_span_seconds"] == 1
    assert public_text("Bearer abc /root/private/thing") == "[credential redacted] [local path]"


def test_report_requires_native_action_link(tmp_path):
    run = tmp_path / "run"
    (run / "episode").mkdir(parents=True)
    (run / "episode/episode.json").write_text(json.dumps(dict(events=[dict(tool="step",
        arguments=dict(observation_id="4"), execution=dict(executed_native_steps=15))])))
    with pytest.raises(ValueError, match="native record"):
        export_task(run, tmp_path / "out", 1)


def test_offline_bundle_embeds_files_and_escapes_script_end(tmp_path):
    root = tmp_path / "site"
    (root / "fonts").mkdir(parents=True)
    (root / "fonts/NotoSansSC.ttf").write_bytes(b"font")
    (root / "index.html").write_text('<meta charset="utf-8"><link rel="stylesheet" href="style.css">'
                                    '<script src="app.js" defer></script><body></body>')
    (root / "style.css").write_text('@font-face{src:url("fonts/NotoSansSC.ttf")}')
    (root / "app.js").write_text('document.title = "fixture";')
    (root / "video.mp4").write_bytes(b"video")
    (root / "trace.json").write_text('{}')
    (root / "data.json").write_text(json.dumps(dict(tasks=[dict(videos={"head": "video.mp4"},
        observations=[], download="trace.json", title="</script><script>bad()</script>")])))
    out = tmp_path / "report.html"
    result = bundle(root, out)
    html = out.read_text()
    assert result["embedded_assets"] == 3 and result["tasks"] == 1
    assert 'src="app.js"' not in html and 'href="style.css"' not in html
    assert 'data:font/ttf;base64,' in html and 'connect-src \'none\'' in html
    assert '<script>bad()' not in html
    assert json.loads(script_json({"text": "</script>&\u2028"})) == {"text": "</script>&\u2028"}
    (tmp_path / "outside.mp4").write_bytes(b"private")
    (root / "video.mp4").unlink()
    (root / "video.mp4").symlink_to(tmp_path / "outside.mp4")
    with pytest.raises(ValueError, match="outside reviewed report"):
        bundle(root, out)


def test_readonly_report_server_ranges_and_boundary(tmp_path):
    root = tmp_path / "public"
    root.mkdir()
    (root / "index.html").write_text("hello")
    (root / "video.mp4").write_bytes(b"0123456789")
    (tmp_path / "secret.json").write_text("secret")
    (root / "outside.json").symlink_to(tmp_path / "secret.json")
    (root / "empty").mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with client.open(url + "/") as response:
            assert response.read() == b"hello"
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        for range_, expected in (("bytes=2-5", b"2345"), ("bytes=-3", b"789"), ("bytes=8-", b"89")):
            request = urllib.request.Request(url + "/video.mp4", headers={"Range": range_})
            with client.open(request) as response:
                assert response.status == 206 and response.read() == expected
        for path in ("/outside.json", "/%2e%2e/secret.json", "/empty/", "/auth.json"):
            with pytest.raises(urllib.error.HTTPError) as error:
                client.open(url + path)
            assert error.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as error:
            client.open(urllib.request.Request(url + "/video.mp4", headers={"Range": "bytes=99-"}))
        assert error.value.code == 416
        with pytest.raises(urllib.error.HTTPError) as error:
            client.open(urllib.request.Request(url + "/", data=b"write"))
        assert error.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
