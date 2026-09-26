"""Render the maintained checkpoint catalog as an editable README figure."""
from collections import OrderedDict
import json
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]


def render() -> str:
    models = json.loads((ROOT / "docs/supported-models.json").read_text())["models"]
    families = OrderedDict()
    for model in models:
        families.setdefault(model["family"], OrderedDict()).setdefault(model["series"], []).append(model)
    series_count = sum(len(series) for series in families.values())
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 1040" width="1120" height="1040">',
        "<title>JevAny supported model families</title>",
        "<desc>Seven families, sixteen series and thirty-seven official checkpoints. "
        "Each size selects one main checkpoint; quantized and Base variants are not separate entries.</desc>",
        '<rect width="1120" height="1040" rx="24" fill="#ffffff"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
    ]

    def text(x, y, value, size=21, color="#202938", weight=400, anchor="start"):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
                     f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')

    text(36, 43, f"{len(families)} families", 27, weight=700)
    text(1084, 43, f"{series_count} series  ·  {len(models)} checkpoints", 22, "#596579", anchor="end")

    def card(family, x, y, width, height, color):
        series = families[family]
        count = sum(len(entries) for entries in series.values())
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" '
                     'fill="#f8fafc" stroke="#e1e6ed"/>')
        parts.append(f'<rect x="{x+22}" y="{y+24}" width="5" height="25" rx="2" fill="{color}"/>')
        text(x + 40, y + 46, family, 27, weight=700)
        noun = "model" if count == 1 else "models"
        text(x + width - 24, y + 45, f"{len(series)} series · {count} {noun}", 17, "#687487", anchor="end")
        return list(series.items())

    def entries(x, y, series, models, line_width=44):
        text(x, y, series, 18, "#596579", 600)
        labels = []
        for model in models:
            label = model["size"]
            if model["id"].endswith("-2507"):
                label += " (2507)"
            labels.append(label)
        lines, line = [], ""
        for label in labels:
            candidate = f"{line}  ·  {label}" if line else label
            if line and len(candidate) > line_width:
                lines.append(line)
                line = label
            else:
                line = candidate
        lines.append(line)
        for index, line in enumerate(lines):
            text(x, y + 29 + 26 * index, line)

    qwen = card("Qwen", 36, 72, 1048, 344, "#7260cc")
    for index, (series, models) in enumerate(qwen):
        entries(60 + (index // 3) * 532, 153 + (index % 3) * 84, series, models, 41)
    for family, x, y, height, color in [
        ("Gemma", 36, 440, 140, "#3875cf"), ("Muse", 572, 440, 140, "#356e9a"),
        ("Mistral", 36, 604, 186, "#cc8136"), ("GLM", 572, 604, 186, "#4d7caa"),
        ("Nemotron", 36, 814, 194, "#67833c"), ("Llama", 572, 814, 194, "#317ac0"),
    ]:
        for index, (series, models) in enumerate(card(family, x, y, 512, height, color)):
            entries(x + 24, y + 80 + index * 66, series, models)
    parts.extend(["</g>", "</svg>"])
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    (ROOT / "docs/supported-model-families.svg").write_text(render())
