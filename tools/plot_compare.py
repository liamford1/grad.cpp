#!/usr/bin/env python3
"""Render an A/B of two training runs to a self-contained SVG.

Top panel: each run's smoothed training loss. Bottom panel: the gap
between them (B minus A) at matching steps, smoothed more heavily, on its own linear axis, so a
difference of a few hundredths of a nat is readable instead of lost in the
top panel's log scale. Standard library only.

    python3 tools/plot_compare.py a_metrics.csv b_metrics.csv out.svg \\
        --labels "GPT-2 block" "Llama block"
"""

import argparse
import math
from pathlib import Path

from plot_run import EMA_ALPHA, read_metrics

WIDTH, HEIGHT = 820, 560
LEFT, RIGHT = 64, 170
TOP_PANEL = (56, 330)      # y range of the loss panel
BOTTOM_PANEL = (390, 510)  # y range of the gap panel
LOSS_TICKS = [1.5, 2, 3, 4]
MAX_POINTS = 800
GAP_FROM_STEP = 2000   # before this the EMAs are still converging from step 0
GAP_ALPHA = 0.002      # heavier smoothing: the per-step gap is noisy once batches differ

STYLE = """
  .surface { fill: #fcfcfb; }
  .title { fill: #0b0b0b; font: 600 15px system-ui, -apple-system, sans-serif; }
  .subtitle { fill: #0b0b0b; font: 600 13px system-ui, -apple-system, sans-serif; }
  .label { fill: #52514e; font: 12px system-ui, -apple-system, sans-serif; }
  .grid { stroke: #e4e3df; stroke-width: 1; }
  .axis { stroke: #b9b8b3; stroke-width: 1; }
  .a { stroke: #2a78d6; }
  .b { stroke: #eb6834; }
  .gap { stroke: #52514e; }
  .swatch-a { fill: #2a78d6; }
  .swatch-b { fill: #eb6834; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .title, .subtitle { fill: #ffffff; }
    .label { fill: #c3c2b7; }
    .grid { stroke: #2e2e2c; }
    .axis { stroke: #5c5b57; }
    .a { stroke: #3987e5; }
    .b { stroke: #d95926; }
    .gap { stroke: #c3c2b7; }
    .swatch-a { fill: #3987e5; }
    .swatch-b { fill: #d95926; }
  }
"""


def ema_by_step(points):
    out, ema = {}, None
    for step, loss in points:
        ema = loss if ema is None else EMA_ALPHA * loss + (1 - EMA_ALPHA) * ema
        out[step] = ema
    return out


def thin(points):
    stride = max(1, len(points) // MAX_POINTS)
    return points[::stride] + ([points[-1]] if len(points) % stride else [])


def render(a, b, raw_a, raw_b, labels, title, first_step):
    common = sorted(set(a) & set(b))
    common = [s for s in common if s >= first_step]
    x_min, x_max = common[0], common[-1]
    plot_w = WIDTH - LEFT - RIGHT

    def x(step):
        return LEFT + plot_w * (step - x_min) / (x_max - x_min)

    top, bottom = TOP_PANEL
    lo, hi = math.log(1.45), math.log(LOSS_TICKS[-1])

    def y_loss(v):
        t = (math.log(min(max(v, 1.45), LOSS_TICKS[-1])) - lo) / (hi - lo)
        return top + (bottom - top) * (1 - t)

    gaps, g = [], None
    for s in common:
        raw = raw_b[s] - raw_a[s]
        g = raw if g is None else GAP_ALPHA * raw + (1 - GAP_ALPHA) * g
        if s >= GAP_FROM_STEP:
            gaps.append((s, g))
    g_lo = min(-0.05, math.floor(min(v for _, v in gaps) * 20) / 20)
    g_ticks = [round(g_lo + i * 0.05, 2) for i in range(int(round(-g_lo / 0.05)) + 1)]
    g_top, g_bottom = BOTTOM_PANEL

    def y_gap(v):
        return g_top + (g_bottom - g_top) * (0 - v) / (0 - g_lo)

    def path(points, fy):
        return "M" + " L".join(f"{x(s):.1f},{fy(v):.1f}" for s, v in points)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="{title}">',
        f"<style>{STYLE}</style>",
        f'<rect class="surface" width="{WIDTH}" height="{HEIGHT}" rx="8"/>',
        f'<text class="title" x="{LEFT}" y="30">{title}</text>',
    ]

    for tick in LOSS_TICKS:
        ty = y_loss(tick)
        parts.append(f'<line class="grid" x1="{LEFT}" x2="{LEFT + plot_w}" y1="{ty:.1f}" y2="{ty:.1f}"/>')
        parts.append(f'<text class="label" x="{LEFT - 10}" y="{ty + 4:.1f}" text-anchor="end">{tick:g}</text>')
    parts.append(f'<text class="label" transform="translate(18 {(top + bottom) / 2:.1f}) rotate(-90)" '
                 f'text-anchor="middle">train loss, EMA (log)</text>')

    for cls, series in (("a", a), ("b", b)):
        pts = thin([(s, series[s]) for s in common])
        parts.append(f'<path class="{cls}" d="{path(pts, y_loss)}" fill="none" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')

    # Direct labels, nudged apart when the final values are close.
    ya, yb = y_loss(a[x_max]), y_loss(b[x_max])
    if abs(ya - yb) < 30:
        mid = (ya + yb) / 2
        ya, yb = (mid - 15, mid + 15) if a[x_max] > b[x_max] else (mid + 15, mid - 15)
    lx = LEFT + plot_w + 10
    for cls, ly, name, value in (("swatch-a", ya, labels[0], a[x_max]),
                                 ("swatch-b", yb, labels[1], b[x_max])):
        parts.append(f'<rect class="{cls}" x="{lx}" y="{ly - 9:.1f}" width="10" height="3" rx="1.5"/>')
        parts.append(f'<text class="label" x="{lx + 16}" y="{ly - 4:.1f}">{name}</text>')
        parts.append(f'<text class="label" x="{lx + 16}" y="{ly + 11:.1f}">{value:.3f}</text>')

    parts.append(f'<text class="subtitle" x="{LEFT}" y="{g_top - 16}">'
                 f'Gap: {labels[1]} minus {labels[0]}, nats (from step {GAP_FROM_STEP // 1000}k)</text>')
    for tick in g_ticks:
        ty = y_gap(tick)
        cls = "axis" if tick == 0 else "grid"
        parts.append(f'<line class="{cls}" x1="{LEFT}" x2="{LEFT + plot_w}" y1="{ty:.1f}" y2="{ty:.1f}"/>')
        parts.append(f'<text class="label" x="{LEFT - 10}" y="{ty + 4:.1f}" text-anchor="end">{tick:+.2f}</text>')
    parts.append(f'<path class="gap" d="{path(thin(gaps), y_gap)}" fill="none" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append(f'<text class="label" x="{lx}" y="{y_gap(gaps[-1][1]) + 4:.1f}">{gaps[-1][1]:+.3f}</text>')

    for s in range(0, x_max + 1, 10000):
        if s >= x_min:
            parts.append(f'<text class="label" x="{x(s):.1f}" y="{g_bottom + 20}" '
                         f'text-anchor="middle">{s // 1000}k</text>')
    parts.append(f'<text class="label" x="{LEFT + plot_w / 2:.1f}" y="{HEIGHT - 12}" '
                 f'text-anchor="middle">optimizer step</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--labels", nargs=2, default=["A", "B"])
    parser.add_argument("--title", default="Training loss A/B")
    parser.add_argument("--from-step", type=int, default=500,
                        help="first step shown; early loss swamps the scale")
    args = parser.parse_args()

    a, _, _ = read_metrics(args.a)
    b, _, _ = read_metrics(args.b)
    if not a or not b:
        parser.error("both CSVs need training rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(ema_by_step(a), ema_by_step(b), dict(a), dict(b), args.labels,
                                  args.title, args.from_step))


if __name__ == "__main__":
    main()
