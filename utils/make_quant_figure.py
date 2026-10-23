"""
make_quant_figure.py

Quantitative-only companion to make_full_results_figure.py: a single figure
with ONE panel per task (bar chart or radar chart, whichever applies), no
structure images. Laid out as 4 rows -- 3 + 3 + 4 + 4 = 14 tasks -- matching
the bar-chart-row-group / radar-chart-row-group split already used for chart
type selection, so no row mixes bar and radar panels.

Reuses TASKS, the color map, and the panel-drawing functions from
make_full_results_figure.py so the two figures stay visually consistent and
in sync with a single source of data.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

plt.rcParams["hatch.linewidth"] = 4.0

from make_full_results_figure import (
    TASKS,
    build_global_color_map,
    get_shown_metrics,
    draw_bar_panel,
    draw_radar_panel,
    draw_placeholder_slot,
    RADAR_MIN_METRICS,
)
import make_full_results_figure as mfrf

# 4 rows: 3 + 3 + 4 + 4 = 14 tasks. Rows 1-2 are the bar-chart tasks
# (PPI / PepPI families); rows 3-4 are the radar-chart tasks (DPI / RPI /
# CarbPI / MPI / SMPI families). No row mixes chart types.
QUANT_ROWS = [
    ["PPI-60",        "PPI-315",        "PPI-70"],
    ["PPI-355",       "PepPI-125",     "PepPI-639"],
    ["DPI-129 (DNA)", "DPI-181 (DNA)",  "RPI (RNA)",       "CarbPI"],
    ["MPI-Ca (Metal)","MPI-Fe (Metal)", "MPI-Mn (Metal)",  "SMPI"],
]


def display_title(task_name):
    """Drop a trailing parenthetical qualifier for the panel title, e.g.
    'DPI-129 (DNA)' -> 'DPI-129', 'RPI (RNA)' -> 'RPI'. The full task_name
    (with qualifier) is still used as the TASKS dict key / lookup."""
    return task_name.split(" (")[0]


# Every panel here gets the full width/height of its own slot (no
# structure image sharing the row), so the panels are much bigger than in
# the combined figure the bar/radar drawing functions were originally
# tuned for. Scale their fixed point-size fonts up to match, otherwise
# text stays the same absolute size while the bars/axes around it grow,
# and ends up looking small relative to the rest of the panel.
QUANT_FONTSIZE_SCALE = 1.6


def draw_quant_panel(fig, sub_gs, col, task_name, tasks_dict):
    task_data = tasks_dict.get(task_name)
    methods_dict = task_data["methods"] if task_data else {}
    title = display_title(task_name)

    if not task_data or not methods_dict:
        ax = fig.add_subplot(sub_gs[0, col])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#FAFAFA")
        ax.text(0.5, 0.5, f"{title}\n(results not yet available)",
                 ha="center", va="center", fontsize=16 * QUANT_FONTSIZE_SCALE,
                 color="gray", transform=ax.transAxes)
        return

    n_common = len(get_shown_metrics(methods_dict, min_methods=len(methods_dict)))
    if n_common >= RADAR_MIN_METRICS:
        ax = fig.add_subplot(sub_gs[0, col], projection="polar")
        draw_radar_panel(ax, methods_dict, title, fontsize_scale=QUANT_FONTSIZE_SCALE)
    else:
        ax = fig.add_subplot(sub_gs[0, col])
        draw_bar_panel(ax, methods_dict, title, fontsize_scale=QUANT_FONTSIZE_SCALE)


def make_figure(rows, tasks_dict, output_path):
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    fig = plt.figure(figsize=(20 * n_cols, 0.6 + 15 * n_rows))

    outer_gs = gridspec.GridSpec(
        n_rows + 1, 1,
        height_ratios=[0.25] + [1] * n_rows,
        hspace=0.45,
    )

    header_ax = fig.add_subplot(outer_gs[0, 0])
    header_ax.axis("off")
    header_ax.text(0.5, 0.6,
                    "Quantitative Comparison on Residue-Level Interaction-Site Prediction",
                    transform=header_ax.transAxes, ha="center", va="center",
                    fontsize=45 * QUANT_FONTSIZE_SCALE, fontweight="bold")

    for row_idx, task_names in enumerate(rows):
        # Each row gets its OWN column count, spanning the full figure
        # width -- so a 3-task row (bar charts) is just as wide overall as
        # a 4-task row (radar charts), rather than leaving an empty slot.
        row_gs = gridspec.GridSpecFromSubplotSpec(
            1, len(task_names), subplot_spec=outer_gs[row_idx + 1, 0], wspace=0.4,
        )
        for i, task_name in enumerate(task_names):
            draw_quant_panel(fig, row_gs, i, task_name, tasks_dict)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    mfrf.GLOBAL_COLOR_MAP = build_global_color_map(TASKS)
    os.makedirs("./visualizations", exist_ok=True)
    make_figure(QUANT_ROWS, TASKS, "./visualizations/quantitative_results_figure.png")