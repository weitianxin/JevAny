#!/usr/bin/env python3
"""Render one saved RAGEN episode as a compact animated GIF."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BACKGROUND = "#070911"
PANEL = "#11182a"
TEXT = "#f7f8ff"
MUTED = "#9ca8c2"
ACCENT = "#2dd4bf"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def frame(environment, step, total):
    image = Image.new("RGB", (720, 500), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 24, 696, 476), radius=24, fill=PANEL, outline="#35415d", width=2)
    draw.text((52, 50), f"Jev Agent  ·  {environment.replace('_', ' ').title()}", font=font(24, True), fill=TEXT)
    draw.text((52, 88), f"Step {step['index'] + 1} of {total}", font=font(15), fill=MUTED)
    observation = str(step["observation"])
    lines = observation.splitlines()
    grid_font = font(30, True)
    y = 140
    for line in lines[:9]:
        draw.text((70, y), line, font=grid_font, fill=TEXT, spacing=8)
        y += 38
    action = step["action_name"]
    confidence = float(step["confidence"])
    draw.rounded_rectangle((420, 142, 660, 260), radius=16, fill="#0b2426", outline=ACCENT, width=2)
    draw.text((444, 166), "NEXT ACTION", font=font(13, True), fill=ACCENT)
    draw.text((444, 195), action, font=font(25, True), fill=TEXT)
    draw.text((444, 230), f"confidence  {confidence:.0%}", font=font(14), fill=MUTED)
    status = "SUCCESS" if step["done"] and step["info"].get("success") else "DECIDING"
    draw.text((52, 438), status, font=font(14, True), fill=ACCENT if status == "SUCCESS" else MUTED)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=int, default=900, help="milliseconds per frame")
    args = parser.parse_args()
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    episode = result["traces"][args.episode]
    steps = episode["steps"]
    if not steps:
        raise ValueError("episode contains no steps")
    frames = [frame(result["environment"], step, len(steps)) for step in steps]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=args.duration,
                   loop=0, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
