"""Render all three packaged replays through the same Playground UI.

Requires Pillow, Playwright and its Chromium browser. Start ``jevany demo``,
then run ``python scripts/render_demo_gifs.py``. This records packaged model
decisions; it does not run inference or change any actions.
"""
import argparse
from bisect import bisect_right
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from PIL import Image
from playwright.sync_api import sync_playwright

CASES = ("doom", "crafter", "arm")
SIZE = (1120, 900)
FRAME_MS = 40
SPEED = 1.5
FORMAT = "JevAny Playground v1; 1120x900; 25fps; 1.5x"


def timeline(replay):
    """Retain each environment's frame timing and inter-decision pause."""
    entries, starts, elapsed = [], [], 0.
    steps = replay["steps"]
    entries.append((0, steps[0]["frames"][-1]))
    starts.append(elapsed)
    elapsed += FRAME_MS
    for index, step in enumerate(steps[1:], 1):
        if index > 1:
            elapsed += step.get("step_pause_ms", 950) / SPEED
        for frame in step["frames"]:
            entries.append((index, frame))
            starts.append(elapsed)
            elapsed += step.get("frame_duration_ms", 65) / SPEED
    samples = []
    for time_ms in range(0, math.ceil(elapsed), FRAME_MS):
        entry = entries[bisect_right(starts, time_ms) - 1]
        if samples and samples[-1][0] == entry:
            samples[-1][1] += FRAME_MS
        else:
            samples.append([entry, FRAME_MS])
    samples[0][1] = samples[-1][1] = FRAME_MS
    return samples


def capture(page, case, replay, out):
    page.evaluate("""async key => {
        await selectCase(key); playing = false; loop++;
        index = 0; await show(replay.steps[0]);
    }""", case)
    heading = page.locator(".workspace-heading").bounding_box()
    width = math.ceil(heading["width"] + 32)
    clip = {"x": math.floor(heading["x"] - 16), "y": math.floor(heading["y"] - 16),
            "width": width, "height": round(width * SIZE[1] / SIZE[0])}
    frames, durations = [], []
    previous_step = None
    for (step, uri), duration in timeline(replay):
        if step != previous_step:
            page.evaluate("""async step => {index = step; await show(replay.steps[step]);}""", step)
            workspace = page.locator(".workspace").bounding_box()
            assert workspace["y"] + workspace["height"] <= clip["y"] + clip["height"], (
                case, step, "Playground content exceeds the shared crop")
            previous_step = step
        page.evaluate("""async uri => {
            const scene = document.querySelector('#scene');
            scene.src = uri; await scene.decode();
            await new Promise(requestAnimationFrame);
        }""", uri)
        image = Image.open(io.BytesIO(page.screenshot(clip=clip, full_page=True)))
        frames.append(image.convert("RGB").resize(SIZE, Image.Resampling.LANCZOS))
        durations.append(duration)
    assert page.locator("#error").inner_text() == ""
    assert page.locator("#feedback-label").inner_text() == "GOAL COMPLETED"

    # One palette for the whole episode prevents frame-to-frame colour flicker.
    swatches = Image.new("RGB", (6 * 280, 4 * 225))
    for index in range(24):
        frame = frames[round(index * (len(frames) - 1) / 23)].resize((280, 225))
        swatches.paste(frame, ((index % 6) * 280, (index // 6) * 225))
    palette = swatches.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    images = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    target = out / f"playground-{case}.gif"
    images[0].save(target, save_all=True, append_images=images[1:], duration=durations,
                   loop=0, disposal=1, optimize=True,
                   comment=(FORMAT + "; actual browser replay; " + replay["note"]).encode())
    return {"case": case, "size": list(SIZE), "duration_ms": sum(durations),
            "samples": len(frames), "decisions": len(replay["steps"]) - 1,
            "controller": replay["controller"], "success": replay["steps"][-1]["success"],
            "clip": clip, "bytes": target.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument("--out", type=Path, default=Path("docs/demos"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    errors = []
    with TemporaryDirectory(prefix=".playground-", dir=args.out) as directory:
        staging = Path(directory)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--disable-dev-shm-usage"])
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(args.base_url, wait_until="networkidle")
                results = []
                for case in CASES:
                    with urlopen(f"{args.base_url}/recordings/{case}/replay.json") as response:
                        replay = json.load(response)
                    results.append(capture(page, case, replay, staging))
                    print(json.dumps(results[-1]), flush=True)
            finally:
                browser.close()
        assert not errors, errors
        assert all(item["clip"] == results[0]["clip"] for item in results)
        (staging / "playground-format.json").write_text(
            json.dumps({"format": FORMAT, "cases": results}, indent=2) + "\n")
        for path in staging.iterdir():
            path.replace(args.out / path.name)


if __name__ == "__main__":
    main()
