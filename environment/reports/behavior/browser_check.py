"""Desktop/mobile review smoke; requires isolated Playwright installation."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-gpu"])
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        errors = []
        external_requests = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        if args.offline:
            page.context.set_offline(True)
            page.on("request", lambda request: external_requests.append(request.url)
                    if request.url.startswith(("http:", "https:")) else None)
        page.goto(args.url, wait_until="networkidle")
        page.locator("#workspace").wait_for(state="visible")
        assert page.locator(".task-tab").count() == 4
        for task in range(1, 5):
            page.locator(f'.task-tab[data-id="{task}"]').click()
            page.wait_for_function("id => document.querySelector('.task-tab[aria-selected=true]').dataset.id === String(id)", arg=task)
            page.wait_for_function("() => document.querySelector('#video').readyState >= 2")
            page.locator("#outline button").last.click()
            page.wait_for_timeout(200)
            assert page.locator("#event-detail .gallery img").count() == 3
            assert page.evaluate("[...document.querySelectorAll('.gallery img')].every(i => i.complete && i.naturalWidth > 0)")
        page.locator('.task-tab[data-id="2"]').click()
        page.wait_for_function("() => document.querySelector('.task-tab[aria-selected=true]').dataset.id === '2'")
        page.locator("#filter").select_option("unreached")
        assert page.locator("#outline button").count() == 22
        page.locator("#outline button").first.click()
        assert "EEF 未到位" in page.locator("#event-detail").inner_text()
        page.locator("#search").fill("no-such-event-xyz")
        assert page.locator("#outline button").count() == 0
        page.locator("#search").fill("")
        page.locator('[data-camera="right_wrist"]').click()
        page.wait_for_function("() => document.querySelector('#video').readyState >= 2")
        assert page.locator("#video").evaluate("v => v.videoWidth") == 128
        page.locator('[data-camera="head_640x480"]').click()
        page.wait_for_function("() => document.querySelector('#video').readyState >= 2 && document.querySelector('#video').videoWidth === 640")
        page.locator("#speed").select_option("0.5")
        page.locator("#video").evaluate("v => v.play()")
        page.wait_for_timeout(600)
        page.locator("#video").evaluate("v => v.pause()")
        assert page.locator("#video").evaluate("v => v.currentTime > 0")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(args.output / "desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator('.task-tab[data-id="3"]').click()
        page.wait_for_function("() => document.querySelector('.task-tab[aria-selected=true]').dataset.id === '3'")
        page.wait_for_function("() => document.querySelector('#video').readyState >= 2")
        page.locator("#video").evaluate("v => v.play()")
        page.wait_for_timeout(250)
        page.locator("#video").evaluate("v => v.pause()")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        with page.expect_download() as download:
            page.locator("#download").click()
        download_path = args.output / "download-check.json"
        download.value.save_as(download_path)
        assert json.loads(download_path.read_text())["id"] == 3
        page.screenshot(path=str(args.output / "mobile.png"), full_page=True)
        assert not errors, errors
        assert not external_requests, external_requests
        (args.output / "result.json").write_text(json.dumps(dict(
            passed=True, offline=args.offline, external_requests=external_requests,
            desktop=[1440, 1100], mobile=[390, 844], tasks=4,
            checks=["all videos decoded", "action seek", "RGB images", "unreached filter", "empty search",
                    "camera switch", "playback", "JSON download", "no horizontal overflow", "no page errors"]), indent=2))
        browser.close()


if __name__ == "__main__":
    main()
