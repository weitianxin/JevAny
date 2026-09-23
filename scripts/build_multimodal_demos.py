#!/usr/bin/env python3
"""Build reproducible synthetic image/video decisions and render evaluated demos."""
import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from jevany.suite import digest, semantic_hash, write_json, write_jsonl


WIDTH, HEIGHT = 960, 540
COLORS = {
    "ink": "#111827", "muted": "#64748b", "paper": "#f8fafc",
    "purple": "#7c3aed", "green": "#10b981", "yellow": "#f59e0b", "red": "#ef4444",
}


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def canvas(title, subtitle):
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 24, WIDTH - 28, HEIGHT - 24), 28, fill="white", outline="#e2e8f0", width=3)
    draw.text((64, 52), title, font=font(34, True), fill=COLORS["ink"])
    draw.text((64, 98), subtitle, font=font(21), fill=COLORS["muted"])
    return image


def status_board():
    image = canvas("Warehouse Control", "Route the inspection team using the live bay status")
    draw = ImageDraw.Draw(image)
    bays = [("BAY A", "NORMAL", COLORS["green"]), ("BAY B", "ALERT", COLORS["red"]), ("BAY C", "CHECK", COLORS["yellow"])]
    for index, (name, status, color) in enumerate(bays):
        left = 66 + index * 288
        draw.rounded_rectangle((left, 166, left + 250, 414), 20, fill="#f8fafc", outline="#cbd5e1", width=3)
        draw.ellipse((left + 78, 204, left + 172, 298), fill=color)
        draw.text((left + 125, 326), name, anchor="mm", font=font(27, True), fill=COLORS["ink"])
        draw.text((left + 125, 372), status, anchor="mm", font=font(24, True), fill=color)
    return image


def route_map():
    image = canvas("Robot Route", "The dashed segment is closed for maintenance")
    draw = ImageDraw.Draw(image)
    points = {"START": (140, 350), "NORTH": (460, 190), "EAST": (790, 350)}
    draw.line((points["START"], points["NORTH"]), fill=COLORS["green"], width=16)
    draw.line((points["START"], points["EAST"]), fill=COLORS["red"], width=16)
    for x in range(270, 680, 46):
        draw.line((x, 340, x + 20, 360), fill="white", width=8)
    for name, (x, y) in points.items():
        draw.ellipse((x - 36, y - 36, x + 36, y + 36), fill=COLORS["purple"] if name == "START" else "#0f172a")
        draw.text((x, y + 62), name, anchor="mm", font=font(24, True), fill=COLORS["ink"])
    draw.text((520, 400), "CLOSED", font=font(27, True), fill=COLORS["red"])
    return image


def quality_panel():
    image = canvas("Quality Inspection", "Send the item with the failed check to manual review")
    draw = ImageDraw.Draw(image)
    rows = [("ITEM 17", "PASS", COLORS["green"]), ("ITEM 28", "FAIL", COLORS["red"]), ("ITEM 39", "PASS", COLORS["green"])]
    for index, (item, result, color) in enumerate(rows):
        top = 164 + index * 98
        draw.rounded_rectangle((110, top, 850, top + 72), 16, fill="#f8fafc", outline="#dbe3ee", width=2)
        draw.text((148, top + 36), item, anchor="lm", font=font(27, True), fill=COLORS["ink"])
        draw.text((790, top + 36), result, anchor="rm", font=font(27, True), fill=color)
    return image


def robot_frames(destination, color):
    frames = []
    start, end = (140, 350), {"North Bay": (480, 185), "East Bay": (790, 350), "West Bay": (250, 185)}[destination]
    for step in range(32):
        t = step / 31
        x = int(start[0] + (end[0] - start[0]) * t)
        y = int(start[1] + (end[1] - start[1]) * t)
        image = canvas("Autonomous Cart", "Watch where the cart finishes its route")
        draw = ImageDraw.Draw(image)
        bays = [("WEST BAY", (250, 185)), ("NORTH BAY", (480, 185)), ("EAST BAY", (790, 350))]
        for name, point in bays:
            px, py = point
            draw.rounded_rectangle((px - 92, py - 48, px + 92, py + 48), 14, fill="#e2e8f0", outline="#94a3b8", width=3)
            draw.text((px, py), name, anchor="mm", font=font(21, True), fill=COLORS["ink"])
        draw.line((start, end), fill="#c4b5fd", width=12)
        draw.rounded_rectangle((x - 45, y - 32, x + 45, y + 32), 12, fill=color, outline="white", width=4)
        draw.text((x, y), "CART", anchor="mm", font=font(17, True), fill="white")
        frames.append(image)
    return frames


def write_video(path, frames, rate=8):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def record(identifier, kind, uri, question, options, label):
    source = f"synthetic_{kind}_decision"
    result = {
        "state": {"demo": "JevAny multimodal decision", "modality": kind},
        "media": [{"type": kind, "uri": uri}],
        "questions": {"answer": {
            "type": "choice", "instructions": question,
            "criteria": dict(zip("abc", options)), "label": label, "src": source,
        }},
    }
    media_hash = digest(Path(uri))
    result["_meta"] = {
        "source": source, "variant": "clean", "id": identifier,
        "group_id": f"{source}/{identifier}", "split": "development",
        "media_sha256": [media_hash], "text_sha256": semantic_hash(result),
    }
    return result


def build(output):
    output.mkdir(parents=True)
    media = output / "media"
    media.mkdir()
    image_specs = [
        ("status", status_board(), "Which bay requires immediate inspection?", ["Bay A", "Bay B", "Bay C"], "b"),
        ("route", route_map(), "Which destination remains reachable from START?", ["North", "East", "Neither"], "a"),
        ("quality", quality_panel(), "Which item should go to manual review?", ["Item 17", "Item 28", "Item 39"], "b"),
    ]
    rows = []
    for name, image, question, options, label in image_specs:
        path = media / f"image-{name}.png"
        image.save(path, optimize=True)
        rows.append(record(f"demo-image-{name}", "image", str(path.resolve()), question, options, label))
    video_specs = [("north", "North Bay", COLORS["purple"], "b"), ("east", "East Bay", COLORS["green"], "c"), ("west", "West Bay", COLORS["yellow"], "a")]
    for name, destination, color, label in video_specs:
        path = media / f"video-{name}.mp4"
        write_video(path, robot_frames(destination, color))
        rows.append(record(
            f"demo-video-{name}", "video", str(path.resolve()),
            "Where does the cart finish?", ["West Bay", "North Bay", "East Bay"], label,
        ))
    split = output / "development.jsonl"
    write_jsonl(split, rows)
    write_json(output / "manifest.json", {
        "name": output.name,
        "description": "Self-created synthetic media for reproducible JevAny demos",
        "files": {"development.jsonl": {"records": len(rows), "questions": len(rows), "sha256": digest(split)}},
        "base_revisions": {}, "licenses": {"synthetic media": "Apache-2.0"},
        "trainable_sources": [], "eval_only_sources": sorted({r["_meta"]["source"] for r in rows}),
        "holdout_sources": [],
    })
    print(f"wrote {len(rows)} synthetic decisions to {output}")


def load_predictions(path):
    return {row["id"]: row for row in (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines())}


def decision_panel(image, question, options, label, probabilities):
    base = image.resize((640, 360))
    card = Image.new("RGB", (960, 540), "#070b14")
    card.paste(base, (24, 88))
    draw = ImageDraw.Draw(card)
    draw.text((24, 24), "JevAny  ·  Multimodal Decision", font=font(28, True), fill="white")
    draw.multiline_text((688, 90), textwrap.fill(question, width=24), font=font(19, True), fill="white", spacing=6)
    for index, (key, option) in enumerate(zip("abc", options)):
        top = 160 + index * 86
        probability = probabilities[key]
        selected = key == max(probabilities, key=probabilities.get)
        color = COLORS["purple"] if selected else "#263247"
        draw.rounded_rectangle((688, top, 930, top + 62), 12, fill=color)
        draw.text((708, top + 31), option, anchor="lm", font=font(19, selected), fill="white")
        draw.text((908, top + 31), f"{probability:.0%}", anchor="rm", font=font(18, True), fill="white")
    answer = options["abc".index(label)]
    draw.text((24, 486), f"Reference: {answer}", font=font(20, True), fill="#a7f3d0")
    return card


def render(suite, predictions_path, docs):
    rows = [json.loads(line) for line in (Path(suite) / "development.jsonl").read_text(encoding="utf-8").splitlines()]
    predictions = load_predictions(predictions_path)
    selected, summary = {}, []
    for kind in ("image", "video"):
        candidates = [row for row in rows if row["media"][0]["type"] == kind]
        evaluated = []
        for row in candidates:
            probs = predictions[row["_meta"]["id"]]["prediction"]["probabilities"]["answer"]
            label = row["questions"]["answer"]["label"]
            correct = max(probs, key=probs.get) == label
            evaluated.append((correct, probs[label], row, probs))
        correct = sorted((item for item in evaluated if item[0]), key=lambda item: item[1])
        success = correct[len(correct) // 2] if correct else max(evaluated, key=lambda item: item[1])
        failures = [item for item in evaluated if not item[0]]
        alternate = max(failures, key=lambda item: max(item[3].values())) if failures else min(evaluated, key=lambda item: item[1])
        selected[kind] = {"success": success, "alternate": alternate}
        summary.extend({
            "id": item[2]["_meta"]["id"], "modality": kind, "correct": item[0],
            "reference_probability": item[1], "probabilities": item[3],
        } for item in evaluated)
    docs.mkdir(parents=True, exist_ok=True)
    for role, (_, _, row, probs) in selected["image"].items():
        image = Image.open(row["media"][0]["uri"]).convert("RGB")
        question = row["questions"]["answer"]
        decision_panel(image, question["instructions"], list(question["criteria"].values()), question["label"], probs).save(
            docs / f"jev-image-{role}.png", optimize=True,
        )
    for role, (_, _, row, probs) in selected["video"].items():
        question = row["questions"]["answer"]
        with av.open(row["media"][0]["uri"]) as container:
            frames = [frame.to_image() for frame in container.decode(video=0)]
        cards = [decision_panel(frame, question["instructions"], list(question["criteria"].values()), question["label"], probs) for frame in frames]
        cards[0].save(docs / f"jev-video-{role}.gif", save_all=True, append_images=cards[1:], duration=125, loop=0, optimize=True)
    write_json(docs / "multimodal-demo.json", {
        "media": "self-created synthetic examples released under Apache-2.0",
        "selection": {
            "success": "median reference-label probability among correct results from three predeclared examples",
            "alternate": "highest-confidence error, or lowest reference-label probability when all three are correct",
        },
        "selected": {
            kind: {role: item[2]["_meta"]["id"] for role, item in values.items()}
            for kind, values in selected.items()
        },
        "results": summary,
    })
    print(f"rendered image and video demos to {docs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/evals/synthetic-multimodal-demo-v1")
    parser.add_argument("--predictions")
    parser.add_argument("--docs", default="docs/demos")
    args = parser.parse_args()
    if args.predictions:
        render(args.out, args.predictions, Path(args.docs))
    else:
        build(Path(args.out))


if __name__ == "__main__":
    main()
