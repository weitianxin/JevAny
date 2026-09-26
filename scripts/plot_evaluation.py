#!/usr/bin/env python3
"""Render the README checkpoint chart from release results (requires matplotlib)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"SFT": "#328E83", "RLCR": "#9873C3"}


def main():
    release = json.loads((ROOT / "results/release-v0.2.json").read_text())
    image = json.loads((ROOT / "results/multimodal-image-v1.json").read_text())
    video = json.loads((ROOT / "results/multimodal-video-v1.json").read_text())
    models = release["models"]
    checkpoints = [
        models["jevany-27b-sft-v2"],
        models["jevany-27b-rlcr-v2-step1000"],
    ]
    panels = [
        ("Transfer", f'{checkpoints[0]["transfer_v9"]["questions"]:,} questions',
         [model["transfer_v9"]["accuracy"] for model in checkpoints]),
        ("MMLU-Pro", f'{release["test_time_training"]["mmlu_pro"]["questions"]}-question subset',
         [model["transfer_v9"]["mmlu_pro_accuracy"] for model in checkpoints]),
        ("AI2D", "Held-out subset",
         [model["multimodal_holdouts"]["ai2d_accuracy"] for model in checkpoints]),
        ("MMMU", "Held-out subset",
         [model["multimodal_holdouts"]["mmmu_accuracy"] for model in checkpoints]),
        ("MMStar", f'{image["overall"]["questions"]:,} after filtering',
         [image["overall"]["accuracy"]["full"], None]),
        ("MVBench", f'{len(video["tasks"])} tasks / {video["overall"]["questions"]} questions',
         [video["overall"]["accuracy"]["full"], None]),
    ]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "text.color": "#273449",
        "svg.fonttype": "none",
        "svg.hashsalt": "jevany-evaluation-v0.2",
    })
    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor="white")
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.25, top=0.78)
    fig.text(0.045, 0.91, "JevAny-27B", fontsize=23, weight="bold")
    fig.text(0.045, 0.852, "Released checkpoints · v0.2", fontsize=12, color="#667085")
    fig.legend(
        handles=[Patch(facecolor=color, label=name) for name, color in COLORS.items()],
        loc="upper right", bbox_to_anchor=(0.98, 0.945), ncol=2,
        frameon=False, fontsize=13, handlelength=1.1, handleheight=1.1,
        columnspacing=1.8,
    )

    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, len(panels) - 0.4)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", length=0, pad=9, labelsize=11, labelcolor="#667085")
    ax.tick_params(axis="x", length=0, pad=13)
    ax.set_ylabel("Accuracy (%)", labelpad=10, fontsize=12, color="#667085")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(0, color="#AEB8C5", linewidth=1)

    width, offset = 0.30, 0.18
    descriptions = []
    for index, (name, scope, scores) in enumerate(panels):
        values = []
        for position, (checkpoint, color) in enumerate(COLORS.items()):
            x = index + (-offset if position == 0 else offset)
            score = scores[position]
            if score is None:
                ax.text(x, 3, "n/a", ha="center", color="#8892A1", fontsize=11)
                values.append(f"{checkpoint} not evaluated")
                continue
            value = score * 100
            label = f"{value:.2f}" if name == "Transfer" else f"{value:.1f}"
            ax.bar(x, value, width=width, color=color, zorder=3)
            ax.text(x, value + 2.5, label, ha="center", va="bottom",
                    fontsize=11.5, weight="bold")
            values.append(f"{checkpoint} {label}%")
        ax.text(index, -0.155, scope, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9.5, color="#667085")
        descriptions.append(f'{name} ({scope}): {", ".join(values)}.')

    ax.set_xticks(range(len(panels)), [panel[0] for panel in panels],
                  fontsize=12.5, weight="bold")
    fig.text(0.065, 0.07, "n/a = not evaluated", fontsize=10.5, color="#667085")
    fig.savefig(
        ROOT / "docs/evaluation-checkpoints.svg",
        metadata={
            "Date": None,
            "Title": "JevAny-27B v0.2 checkpoint accuracy",
            "Description": " ".join(descriptions),
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
