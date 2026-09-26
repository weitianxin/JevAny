"""Render the maintained checkpoint catalog as an editable README figure."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
LOGOS = ROOT / "docs/model-logos"
ET.register_namespace("", "http://www.w3.org/2000/svg")


def logo(family: str, source: str, x: int, y: int) -> str:
    """Embed the original vector, with unique gradient IDs for each card."""
    tree = ET.fromstring((LOGOS / source).read_text())
    identifiers = {node.attrib["id"]: f"{family.lower()}-{node.attrib['id']}"
                   for node in tree.iter() if "id" in node.attrib}
    for node in tree.iter():
        for name, value in node.attrib.items():
            if name == "id":
                node.set(name, identifiers[value])
            else:
                for original, replacement in identifiers.items():
                    value = value.replace(f"url(#{original})", f"url(#{replacement})")
                node.set(name, value)
    tree.attrib.pop("style", None)
    tree.attrib.update(x=str(x), y=str(y), width="36", height="36", color="#202938")
    return ET.tostring(tree, encoding="unicode")


def render() -> str:
    models = json.loads((ROOT / "docs/supported-models.json").read_text())["models"]
    logos = json.loads((LOGOS / "sources.json").read_text())["families"]
    families = {}
    for model in models:
        families.setdefault(model["family"], []).append(model)

    def card_height(count, columns=1):
        return 118 + (math.ceil(count / columns) - 1) * 34

    layout = []
    y = 72
    qwen_height = card_height(len(families["Qwen"]), 2)
    layout.append(("Qwen", 36, y, 1048, qwen_height, 2))
    y += qwen_height + 24
    other_families = [family for family in families if family != "Qwen"]
    for index in range(0, len(other_families), 2):
        row = other_families[index:index + 2]
        height = max(card_height(len(families[family])) for family in row)
        for column, family in enumerate(row):
            layout.append((family, 36 + column * 536, y, 512, height, 1))
        y += height + 24
    height = y + 12
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 {height}" width="1120" height="{height}">',
        "<title>JevAny supported model families</title>",
        f"<desc>{len(families)} families and {len(models)} official checkpoints, "
        "listed individually with their family logos.</desc>",
        f'<rect width="1120" height="{height}" rx="24" fill="#ffffff"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
    ]

    def text(x, y, value, size=21, color="#202938", weight=400, anchor="start"):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
                     f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')

    text(36, 43, f"{len(families)} families", 27, weight=700)
    text(1084, 43, f"{len(models)} models", 22, "#596579", anchor="end")

    for family, x, y, width, height, columns in layout:
        entries = families[family]
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" '
                     'fill="#f8fafc" stroke="#e1e6ed"/>')
        parts.append(logo(family, logos[family], x + 24, y + 18))
        text(x + 74, y + 46, family, 27, weight=700)
        noun = "model" if len(entries) == 1 else "models"
        text(x + width - 24, y + 45, f"{len(entries)} {noun}", 17, "#687487", anchor="end")
        rows = math.ceil(len(entries) / columns)
        for index, model in enumerate(entries):
            column, row = divmod(index, rows)
            text(x + 24 + column * 536, y + 90 + row * 34, model["id"].split("/", 1)[1], 19)
    parts.extend(["</g>", "</svg>"])
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    (ROOT / "docs/supported-model-families.svg").write_text(render())
