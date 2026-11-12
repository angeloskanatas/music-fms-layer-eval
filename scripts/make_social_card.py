"""Draws the site's link-preview card (1200 x 630 PNG) from the per-layer results in data/results.

The curves show how each task family's performance moves with depth in the five masked-prediction encoders of the
paper: every task's MLP-probe score per layer is scaled from its worst layer (0) to its best layer (1), placed on a depth
axis from the first to the last layer (shape-preserving interpolation between layers), averaged per family, and lightly
smoothed for display. Each dot marks the highest point of its family's curve. The
order (tonal, then semantic, then rhythm) is the paper's Sec. 4 finding; the paper reports the mean depth of the best
layer (~30%, ~53%, ~64%), which a few late best layers pull later than the curves' peaks.

The text uses Inter, the site's font (static TTFs from https://github.com/rsms/inter, extras/ttf); without --fonts the
default matplotlib font is used.

Usage, from the repository root:
    python scripts/make_social_card.py --fonts <folder with Inter-*.ttf>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

W, H = 1200, 630
INK, INK2, INK3, LINE = "#1e293b", "#64748b", "#94a3b8", "#e2e8f0"
ISMIR_GOLD = "#b47500"  # the ISMIR 2026 logo and website colour
MASKED_MODELS = ["mert-v1-95M", "mert-v1-330M", "musicfm", "muq", "omar-rq-multifeature-25hz"]
FAMILIES = [  # (label, tasks with one score each, colour, label offset from the dot in points, horizontal alignment)
    ("Tonal", ["key", "nsynth_pitch", "chords_ace_root"], "#2a78d6", (-12, 8), "right"),
    ("Semantic", ["genre", "emo_r2", "mtt_ap", "mtg_genre_ap", "mtg_instrument_ap", "mtg_mood_ap", "mtg_top50_ap"],
     "#eb6834", (0, 16), "center"),
    ("Rhythm", ["beat_f1", "downbeat_f1"], "#1baf7a", (10, 16), "left"),
]
GRID = np.linspace(0, 1, 401)


def family_profiles(results: Path, tasks: list[str]) -> list[np.ndarray]:
    """The per-layer MLP-probe scores of the family's tasks in every masked encoder."""
    out = []
    for model in MASKED_MODELS:
        records = json.loads((results / model / "downstream.json").read_text())["records"]
        profiles = {r["task"]: r["layers"] for r in records if r["readout"].startswith("mlp") and r["layers"]}
        for task in tasks:
            values = profiles.get(task)
            if values is None or None in values:
                continue
            out.append(np.asarray(values, float))
    return out


def family_curve(profiles: list[np.ndarray]) -> np.ndarray:
    """Mean of the per-task scaled layer profiles on GRID, lightly smoothed."""
    curves = [PchipInterpolator(np.linspace(0, 1, len(v)), (v - v.min()) / (v.max() - v.min()))(GRID) for v in profiles]
    return gaussian_filter1d(np.mean(curves, axis=0), sigma=10, mode="nearest")  # sigma = 2.5% of depth


def px(x: float, y: float) -> tuple[float, float]:
    """Pixel position from the top-left corner -> figure fraction."""
    return x / W, 1 - y / H


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fonts", type=Path, help="folder with the Inter TTF files")
    parser.add_argument("--out", type=Path, default=root / "static/images/social_card.png")
    args = parser.parse_args()
    if args.fonts:
        for font in args.fonts.glob("Inter-*.ttf"):
            font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "Inter"

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("white")

    # text: venue, title, subtitle, address
    fig.text(*px(80, 196), "ISMIR 2026", color=ISMIR_GOLD, fontsize=17, fontweight="semibold", va="baseline")
    fig.text(*px(76, 278), "What Makes a", color=INK, fontsize=52, fontweight="extra bold", va="baseline")
    fig.text(*px(76, 348), "Good Layer?", color=INK, fontsize=52, fontweight="extra bold", va="baseline")
    fig.text(*px(80, 408), "Layer-wise analysis of", color=INK2, fontsize=21, va="baseline")
    fig.text(*px(80, 442), "music foundation models", color=INK2, fontsize=21, va="baseline")
    fig.text(*px(80, 566), "angeloskanatas.github.io/music-fms-layer-eval", color=INK3, fontsize=13,
             fontweight="medium", va="baseline")

    # curves: where each task family peaks across depth
    left, right, top, bottom = 668, 1124, 196, 470
    ax = fig.add_axes([*px(left, bottom), (right - left) / W, (bottom - top) / H])
    ax.set_facecolor("none")
    for label, tasks, colour, offset, ha in FAMILIES:
        profiles = family_profiles(root / "data/results", tasks)
        curve = family_curve(profiles)
        at = int(curve.argmax())
        ax.plot(GRID, curve, color=colour, linewidth=5, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.scatter([GRID[at]], [curve[at]], s=240, color=colour, edgecolor="white", linewidth=3.5, zorder=4,
                   clip_on=False)
        ax.annotate(label, (GRID[at], curve[at]), xytext=offset, textcoords="offset points", ha=ha,
                    va="bottom", color=INK, fontsize=16, fontweight="semibold", zorder=5, annotation_clip=False)
        print(f"{label}: curve peaks at {GRID[at]:.0%} depth ({len(profiles)} task profiles)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.axhline(0, color=LINE, linewidth=1.5, zorder=2)
    ax.axis("off")
    for x, text, ha in [(0, "first layer", "left"), (1, "last layer", "right")]:
        ax.annotate(text, (x, 0), xytext=(0, -14), textcoords="offset points", ha=ha, va="top", color=INK3,
                    fontsize=12.5, annotation_clip=False)

    fig.savefig(args.out, dpi=100, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
