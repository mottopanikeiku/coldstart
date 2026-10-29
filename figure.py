"""Draw the README figure from the committed summary: one dot per seed, bars at medians.

    uv run python figure.py

The design follows the write-up's figure; this standalone SVG carries its own light or dark
palette because GitHub does not load the site's stylesheet.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from control_clock.protocol import THRESHOLDS

# The README table's method names.
NAMES = {
    "ars": "ARS, affine policies",
    "cem": "CEM, affine policies",
    "ppo-cpu": "Batched CPU PPO",
    "sb3-zoo": "SB3 PPO, Zoo settings",
    "cleanrl": "CleanRL PPO, adapted",
}
SHORT = {
    "ars": "ARS",
    "cem": "CEM",
    "ppo-cpu": "batched PPO",
    "sb3-zoo": "SB3 PPO",
    "cleanrl": "CleanRL PPO",
}
TASKS = ("CartPole-v1", "Acrobot-v1")
# The write-up's palette; GitHub draws the README on white or near-black.
THEMES = {
    "light": {"ink": "#1c1b18", "muted": "#69655d", "accent": "#9a3b22", "rule": "#dcd8ce"},
    "dark": {"ink": "#e8e5de", "muted": "#9d988e", "accent": "#e08f72", "rule": "#2f2d29"},
}
TICKS = (0.5, 1, 2, 5, 10, 20, 50, 100)
LOW, HIGH = 0.3, 120


def number(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def draw(summary: dict, theme: dict) -> str:
    groups = []
    for task in TASKS:
        rows = [
            dict(cohort, method=key.split("/")[1])
            for key, cohort in summary.items()
            if key.split("/")[0] == task
        ]
        groups.append((task, sorted(rows, key=lambda row: row["conditional_median_seconds"])))
    seconds = [r["seconds"] for _, rows in groups for row in rows for r in row["records"]]
    if min(seconds) < LOW or max(seconds) > HIGH:
        raise ValueError(f"seed times {min(seconds)}-{max(seconds)} s leave the axis")
    width, left, right, band, top = 560, 168, 52, 22, 44
    bottom = top + sum(24 + band * len(rows) + 8 for _, rows in groups)
    height = bottom + 40

    def x(value):
        span = math.log10(HIGH) - math.log10(LOW)
        return left + (math.log10(value) - math.log10(LOW)) / span * (width - right - left)

    parts = []

    def text(px, py, content, cls="", anchor="start"):
        c = f' class="{cls}"' if cls else ""
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        parts.append(f'<text{c}{a} x="{number(px)}" y="{number(py)}">{html.escape(content)}</text>')

    def line(x1, y1, x2, y2, cls):
        parts.append(
            f'<line class="{cls}" x1="{number(x1)}" y1="{number(y1)}" '
            f'x2="{number(x2)}" y2="{number(y2)}"/>'
        )

    text(left, 16, "one dot per seed; the bar is the median")
    text(width, 16, "passed", anchor="end")
    y = top
    for task, rows in groups:
        threshold = number(THRESHOLDS[task]).replace("-", "\u2212")
        text(0, y, f"{task}, pass at mean return {threshold}", "strong")
        y += 24
        for tick in TICKS:
            line(x(tick), y - 14, x(tick), y + band * len(rows) - 12, "grid")
        for row in rows:
            cls = "em" if row["method"] == "ppo-cpu" else ""
            text(left - 12, y, NAMES[row["method"]], cls, "end")
            for record in row["records"]:
                dot = "dot" if record["solved"] else "dot hollow"
                dot = f"{dot} {cls}".strip()
                cy = y - 4 + (record["seed"] % 5 - 2) * 1.8
                parts.append(
                    f'<circle class="{dot}" cx="{number(x(record["seconds"]))}" '
                    f'cy="{number(cy)}" r="2.3"/>'
                )
            median = row["conditional_median_seconds"]
            line(x(median), y - 12, x(median), y + 4, "mean")
            text(width, y, f"{row['successes']}/{row['seeds']}", anchor="end")
            y += band
        y += 8
    for tick in TICKS:
        text(x(tick), bottom + 6, f"{tick:g}", anchor="middle")
    text(
        (left + width - right) / 2,
        height - 6,
        "seconds from process start, log scale",
        "",
        "middle",
    )

    passed = sum(row["successes"] for _, rows in groups for row in rows)
    runs = sum(row["seeds"] for _, rows in groups for row in rows)
    medians = " ".join(
        f"{task}: "
        + "; ".join(
            f"{SHORT[row['method']]} {row['conditional_median_seconds']:.2f} s" for row in rows
        )
        + "."
        for task, rows in groups
    )
    description = f"Median seconds by method. {medians} {passed} of {runs} runs passed."
    style = (
        "text{font-family:Charter,'Bitstream Charter',Cambria,Georgia,serif;font-size:13px;"
        "font-variant-numeric:lining-nums tabular-nums;fill:%(muted)s}"
        "text.strong{fill:%(ink)s}text.em{fill:%(accent)s}"
        ".grid{stroke:%(rule)s;stroke-width:1}.mean{stroke:%(ink)s;stroke-width:2.5}"
        ".dot{fill:%(muted)s}.dot.em{fill:%(accent)s}"
        ".dot.hollow{fill:none;stroke:%(muted)s;stroke-width:1}"
        ".dot.hollow.em{stroke:%(accent)s;stroke-width:1.5}"
    ) % theme
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="title desc">'
        '<title id="title">Seconds from process start to a passing policy, per seed</title>'
        f'<desc id="desc">{html.escape(description)}</desc><style>{style}</style>'
        + "".join(parts)
        + "</svg>\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("docs"))
    args = parser.parse_args()
    summary = json.loads(args.input.read_text())
    for name, theme in THEMES.items():
        destination = args.output / f"figure-{name}.svg"
        destination.write_text(draw(summary, theme))
        print(destination)


if __name__ == "__main__":
    main()
