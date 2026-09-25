#!/usr/bin/env python3
"""Render a training run's loss curves to a self-contained SVG.

Reads the <prefix>_metrics.csv every run writes (see include/grad/utils/metrics.h)
and plots the smoothed training loss, the validation loss, and the points
where the run was resumed from a checkpoint. Standard library only, so it
runs anywhere the repo does.

    python3 tools/plot_run.py tinystories_metrics.csv docs/runs/loss.svg
"""

import argparse
import csv
import math
from pathlib import Path

WIDTH, HEIGHT = 820, 440
MARGIN = {"left": 64, "right": 170, "top": 56, "bottom": 52}
Y_TICKS = [1.5, 2, 3, 4, 6, 10]
EMA_ALPHA = 0.01
MAX_POINTS = 800

STYLE = """
  .surface { fill: #fcfcfb; }
  .title { fill: #0b0b0b; font: 600 15px system-ui, -apple-system, sans-serif; }
  .label { fill: #52514e; font: 12px system-ui, -apple-system, sans-serif; }
  .grid { stroke: #e4e3df; stroke-width: 1; }
  .axis { stroke: #b9b8b3; stroke-width: 1; }
  .resume { stroke: #b9b8b3; stroke-width: 1; stroke-dasharray: 3 4; }
  .train { stroke: #2a78d6; }
  .val { stroke: #eb6834; }
  .val-dot { fill: #eb6834; stroke: #fcfcfb; stroke-width: 2; }
  .swatch-train { fill: #2a78d6; }
  .swatch-val { fill: #eb6834; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .title { fill: #ffffff; }
    .label { fill: #c3c2b7; }
    .grid { stroke: #2e2e2c; }
    .axis, .resume { stroke: #5c5b57; }
    .train { stroke: #3987e5; }
    .val { stroke: #d95926; }
    .val-dot { fill: #d95926; stroke: #1a1a19; }
    .swatch-train { fill: #3987e5; }
    .swatch-val { fill: #d95926; }
  }
"""


def read_metrics(path):
    train, val, resumes = [], [], []
    segment_start = False
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            kind = row[0]
            if kind == "m":
                segment_start = bool(train)
            elif kind == "t":
                step, loss = int(row[1]), float(row[2])
                if segment_start:
                    resumes.append(step)
                    segment_start = False
                train.append((step, loss))
            elif kind == "e":
                val.append((int(row[1]), float(row[2])))
    return train, val, resumes


def smooth(points):
    out, ema = [], None
    for step, loss in points:
        ema = loss if ema is None else EMA_ALPHA * loss + (1 - EMA_ALPHA) * ema
        out.append((step, ema))
    stride = max(1, len(out) // MAX_POINTS)
    return out[::stride] + ([out[-1]] if len(out) % stride else [])


def render(train, val, resumes, title, val_label):
    plot_w = WIDTH - MARGIN["left"] - MARGIN["right"]
    plot_h = HEIGHT - MARGIN["top"] - MARGIN["bottom"]
    x_max = max(s for s, _ in train)
    y_lo, y_hi = math.log(1.3), math.log(Y_TICKS[-1])

    def x(step):
        return MARGIN["left"] + plot_w * step / x_max

    def y(loss):
        t = (math.log(min(max(loss, 1.3), Y_TICKS[-1])) - y_lo) / (y_hi - y_lo)
        return MARGIN["top"] + plot_h * (1 - t)

    def path(points):
        return "M" + " L".join(f"{x(s):.1f},{y(l):.1f}" for s, l in points)

    bottom = MARGIN["top"] + plot_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="{title}">',
        f"<style>{STYLE}</style>",
        f'<rect class="surface" width="{WIDTH}" height="{HEIGHT}" rx="8"/>',
        f'<text class="title" x="{MARGIN["left"]}" y="30">{title}</text>',
    ]

    for tick in Y_TICKS:
        ty = y(tick)
        parts.append(f'<line class="grid" x1="{MARGIN["left"]}" x2="{MARGIN["left"] + plot_w}" '
                     f'y1="{ty:.1f}" y2="{ty:.1f}"/>')
        parts.append(f'<text class="label" x="{MARGIN["left"] - 10}" y="{ty + 4:.1f}" '
                     f'text-anchor="end">{tick:g}</text>')

    step_tick = 10000
    for s in range(0, x_max + 1, step_tick):
        parts.append(f'<text class="label" x="{x(s):.1f}" y="{bottom + 20}" '
                     f'text-anchor="middle">{s // 1000}k</text>')
    parts.append(f'<text class="label" x="{MARGIN["left"] + plot_w / 2:.1f}" y="{HEIGHT - 12}" '
                 f'text-anchor="middle">optimizer step</text>')
    parts.append(f'<text class="label" transform="translate(18 {MARGIN["top"] + plot_h / 2:.1f}) '
                 f'rotate(-90)" text-anchor="middle">cross-entropy loss (log scale)</text>')
    parts.append(f'<line class="axis" x1="{MARGIN["left"]}" x2="{MARGIN["left"] + plot_w}" '
                 f'y1="{bottom}" y2="{bottom}"/>')

    for i, step in enumerate(resumes):
        rx = x(step)
        parts.append(f'<line class="resume" x1="{rx:.1f}" x2="{rx:.1f}" '
                     f'y1="{MARGIN["top"]}" y2="{bottom}"/>')
        if i == 0:
            parts.append(f'<text class="label" x="{rx + 6:.1f}" y="{MARGIN["top"] + 12}">'
                         f'resumed from checkpoint</text>')

    smoothed = smooth(train)
    parts.append(f'<path class="train" d="{path(smoothed)}" fill="none" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append(f'<path class="val" d="{path(val)}" fill="none" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
    last_step, last_val = val[-1]
    parts.append(f'<circle class="val-dot" cx="{x(last_step):.1f}" cy="{y(last_val):.1f}" r="4"/>')

    # Direct labels at the right edge, nudged apart if they would collide.
    label_x = MARGIN["left"] + plot_w + 10
    train_y, val_y = y(smoothed[-1][1]), y(last_val)
    if abs(train_y - val_y) < 30:
        mid = (train_y + val_y) / 2
        train_y, val_y = mid - 15, mid + 15
    for cls, ly, name, value in (("swatch-train", train_y, "train (EMA)", smoothed[-1][1]),
                                 ("swatch-val", val_y, val_label, last_val)):
        parts.append(f'<rect class="{cls}" x="{label_x}" y="{ly - 9:.1f}" width="10" height="3" rx="1.5"/>')
        parts.append(f'<text class="label" x="{label_x + 16}" y="{ly - 4:.1f}">{name}</text>')
        parts.append(f'<text class="label" x="{label_x + 16}" y="{ly + 11:.1f}">{value:.3f}</text>')

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="Training and validation loss")
    parser.add_argument("--val-label", default="validation",
                        help="legend text for the in-loop validation series")
    args = parser.parse_args()

    train, val, resumes = read_metrics(args.metrics)
    if not train or not val:
        parser.error(f"{args.metrics} has no training or validation rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(train, val, resumes, args.title, args.val_label))


if __name__ == "__main__":
    main()
