"""
make_full_results_figure.py

Single combined figure covering all 14 benchmark evaluations. Each row holds
TWO task "slots" side by side, each slot itself a quantitative+qualitative
triple: (1) radar chart if >=4 metrics are shared across methods, otherwise a
bar chart; (2) best-case structure render, front view; (3) best-case
structure render, back view. An "F1 = x.xxx   MCC = x.xxx" caption (PRIME-ISP's
own numbers) is printed below each slot's front/back image pair.

Row packing rule: a task family with multiple test sets (PPI-Train335 with
Test60/Test315; DPI with Test129/Test181) fills both slots of its own row.
Task families with a single test set are packed two-per-row. This keeps the
figure to 7 rows x 2 slots = 14 slots total, instead of 14 stacked rows.
Bar-chart rows are grouped together, followed by radar-chart rows, so chart
type doesn't alternate row to row.

No "(SOTA)" labeling -- every method is presented on equal footing. Multi-task
/ unified frameworks (ALLSites, PRIME-ISP) are visually distinguished from
single-task specialized methods: hatched bars in the bar-chart panels, solid
+ marker lines (vs. thin dashed lines) in the radar panels.

A slot whose task isn't in TASKS yet (data not collected) renders as a
labeled placeholder rather than breaking the grid, so ROWS can be locked in
before every result is in hand.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import os

METRIC_NAMES = ["F1", "MCC", "Precision", "Recall", "AUROC"]

# Methods considered "multi-task" / unified frameworks, visually distinguished
# from single-task specialized methods. Add to this set as needed.
MULTI_TASK_METHODS = {"ALLSites", "PRIME-ISP"}

TASKS = {
    "RPI (RNA)": {
        "methods": {
            "GraphBind":    [0.358, 0.322, 0.294, 0.463, 0.854],
            "ALLSites":     [0.336, 0.303, 0.277, 0.457, 0.853],
            "DeepPANB":     [0.391, 0.354, 0.351, 0.481, 0.871],
            "PRIME-ISP":    [0.374, 0.346, 0.286, 0.541, 0.871],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/RPI-site/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/RPI-site/best_back.png",
    },
    "DPI-129 (DNA)": {
        "methods": {
            "ALLSites":     [0.501, 0.480, 0.401, 0.674, 0.927],
            "EGPDI":        [0.549, 0.522, 0.503, 0.612, 0.941],
            "DeepPANB":     [0.592, 0.564, 0.559, 0.629, 0.945],
            "PRIME-ISP":    [0.548, 0.533, 0.435, 0.740, 0.941],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/129/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/129/best_back.png",
    },
    "DPI-181 (DNA)": {
        "methods": {
            "ALLSites":     [0.389, 0.379, 0.292, 0.595, 0.904],
            "EGPDI":        [0.424, 0.407, 0.346, 0.558, 0.914],
            "DeepPANB":     [0.435, 0.419, 0.407, 0.560, 0.920],
            "PRIME-ISP":    [0.402, 0.402, 0.287, 0.667, 0.917],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/181/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/181/best_back.png",
    },
    "PepPI-639": {
        "methods": {
            "PepBind":           [None,  0.348, None, None, 0.767],
            "ALLSites":          [None,  0.316, None, None, 0.817],
            "DeepProSite":       [None,  0.397, None, None, 0.861],
            "PRIME-ISP":         [0.376, 0.345, 0.290, 0.536, 0.850],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/PepPI-site_Train640-Test639/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/PepPI-site_Train640-Test639/best_back.png",
    },
    "PepPI-125": {
        "methods": {
            "PepBCL":            [None,  0.385, None, None, 0.815],
            "ALLSites":          [None,  0.383, None, None, 0.843],
            "DeepProSite":       [None,  0.451, None, None, 0.883],
            "PRIME-ISP":         [0.4656, 0.4330, 0.4308, 0.5064, 0.8702],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/PepPI-site_Train1154-Test125/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/PepPI-site_Train1154-Test125/best_back.png",
    },
    "PPI-60": {
        "methods": {
            "RGN":               [0.455, 0.344, None, None, 0.779],
            "ALLSites":          [0.448, 0.340, None, None, 0.781],
            "DeepProSite":       [0.470, 0.379, None, None, 0.813],
            "PRIME-ISP":         [0.466, 0.353, 0.397, 0.564, 0.797],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/PPI-site_Train335-Test60-Test315/60/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/PPI-site_Train335-Test60-Test315/60/best_back.png",
    },
    "PPI-315": {
        "methods": {
            "RGN":               [0.390, 0.273, None, None, 0.752],
            "ALLSites":          [0.392, 0.273, None, None, 0.746],
            "DeepProSite":       [0.457, 0.355, None, None, 0.805],
            "PRIME-ISP":         [0.431, 0.323, 0.339, 0.594, 0.789],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/PPI-site_Train335-Test60-Test315/315/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/PPI-site_Train335-Test60-Test315/315/best_back.png",
    },
    "PPI-70": {
        "methods": {
            "EGRET":             [0.438, 0.270, None, None, 0.719],
            "EnsemPPIS":         [0.440, 0.277, None, None, 0.719],
            "ALLSites":          [0.474, 0.319, None, None, 0.755],
            "PRIME-ISP":         [0.4995, 0.3559, 0.3903, 0.6938, 0.7896],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/PPI-site_Train352-Test70/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/PPI-site_Train352-Test70/best_back.png",
    },
    "PPI-355": {
        "methods": {
            "EnsemPPIS":         [0.385, 0.291, None,   None,   0.770],
            "DELPHI":            [0.364, 0.278, None,   None,   0.746],
            "ALLSites":          [0.466, 0.387, None,   None,   0.823],
            "PRIME-ISP":         [0.5445, 0.4965, 0.4124, 0.8013, 0.8965],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/PPI-site_Train9982-Test355/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/PPI-site_Train9982-Test355/best_back.png",
    },
    "MPI-Ca (Metal)": {
        "methods": {
            "MetalBind":       [0.506, 0.527, 0.720, 0.390, 0.952],
            "PRIME-ISP":       [0.671, 0.670, 0.718, 0.629, 0.950],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/MPI-site_CA/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/MPI-site_CA/best_back.png",
    },
    "MPI-Fe (Metal)": {
        "methods": {
            "MetalBind":       [0.676, 0.674, 0.674, 0.679, 0.981],
            "PRIME-ISP":       [0.678, 0.680, 0.611, 0.763, 0.963],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/MPI-site_FE/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/MPI-site_FE/best_back.png",
    },
    "MPI-Mn (Metal)": {
        "methods": {
            "MetalBind":       [0.518, 0.552, 0.615, 0.447, 0.953],
            "PRIME-ISP":       [0.559, 0.558, 0.604, 0.521, 0.948],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/MPI-site_MN/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/MPI-site_MN/best_back.png",
    },
    "CarbPI": {
        "methods": {
            "CAPSIF:G":        [0.543, 0.538, 0.541, 0.590, None],
            "CAPSIF:V":        [0.597, 0.599, 0.598, 0.647, None],
            "ALLSites":        [0.609, 0.603, 0.624, 0.652, None],
            "PRIME-ISP":       [0.665, 0.655, 0.593, 0.757, 0.961],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/CarbPI-site/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/CarbPI-site/best_back.png",
    },
    "SMPI": {
        "methods": {
            "P2Rank":          [0.450, 0.424, 0.597, 0.361, None],
            "ALLSites":        [0.601, 0.560, 0.609, 0.593, None],
            "PRIME-ISP":       [0.664, 0.628, 0.612, 0.726, 0.934],
        },
        "best_case_front_img": "/home/dvnguye2/PRIME_ISP/visualizations/SMPI-site/best_front.png",
        "best_case_back_img":  "/home/dvnguye2/PRIME_ISP/visualizations/SMPI-site/best_back.png",
    },
}

# ======================================================
# Row packing: 7 rows x 2 slots = all 14 evaluations in ONE figure.
# A task family with multiple test sets fills both slots of its own row.
# Single-test-set families are packed two per row. Bar-chart rows are
# grouped together first, then radar-chart rows, so chart type doesn't
# alternate row to row.
# ======================================================
ROWS = [
    # -- Bar-chart rows first (tasks with < RADAR_MIN_METRICS shared metrics) --
    ("PPI-60",         "PPI-315"),        # PPI-Train335 -- both its test sets
    ("PPI-70",         "PPI-355"),        # PPI-Train352 + PPI-Train9982 (singles)
    ("PepPI-125",     "PepPI-639"),      # PepPI's two single-test-set families
    # -- Radar-chart rows next (tasks with >= RADAR_MIN_METRICS shared metrics) --
    ("DPI-129 (DNA)",  "DPI-181 (DNA)"),  # DPI -- both its test sets
    ("RPI (RNA)",      "CarbPI"),         # leftover singles, paired across category
    ("MPI-Ca (Metal)", "MPI-Fe (Metal)"),
    ("MPI-Mn (Metal)", "SMPI"),
]

COLOR_PRIME = "#C0392B"
COLOR_PALETTE = plt.get_cmap("tab20").colors
RADAR_MIN_METRICS = 4


def build_global_color_map(tasks_dict):
    """Assign each method name a fixed color, consistent across every task
    it appears in -- so e.g. ALLSites or GraphBind always look the same
    regardless of which slot/row they're shown in."""
    all_names = set()
    for task_data in tasks_dict.values():
        all_names.update(task_data["methods"].keys())
    all_names.discard("PRIME-ISP")  # always red, handled separately

    color_map = {"PRIME-ISP": COLOR_PRIME}
    for i, name in enumerate(sorted(all_names)):
        color_map[name] = COLOR_PALETTE[i % len(COLOR_PALETTE)]
    return color_map


GLOBAL_COLOR_MAP = None  # populated in __main__ once TASKS is finalized


def get_shown_metrics(methods_dict, min_methods=2):
    keep_idx = []
    for i in range(len(METRIC_NAMES)):
        n_reporting = sum(1 for vals in methods_dict.values() if vals[i] is not None)
        if n_reporting >= min_methods:
            keep_idx.append(i)
    return keep_idx


def is_multi_task(name):
    return name in MULTI_TASK_METHODS


def reorder_methods(methods_dict):
    """Single-task methods first (original order preserved), then all
    multi-task/unified methods grouped together at the end -- so ALLSites
    and PRIME-ISPalways sit adjacent in both bars and legend."""
    single_task = {k: v for k, v in methods_dict.items() if not is_multi_task(k)}
    multi_task = {k: v for k, v in methods_dict.items() if is_multi_task(k)}
    multi_task_ordered = dict(sorted(multi_task.items(),
                                       key=lambda kv: kv[0] == "PRIME-ISP"))
    return {**single_task, **multi_task_ordered}


def draw_bar_panel(ax, methods_dict, title, fontsize_scale=1.0):
    """fontsize_scale lets a caller with much larger per-panel real estate
    (e.g. make_quant_figure.py, one panel per task instead of one panel
    shared with a structure image) scale every label up together, rather
    than the panel growing while the text stays fixed at the same point
    size and ending up looking small relative to the enlarged bars."""
    s = fontsize_scale
    methods_dict = reorder_methods(methods_dict)
    keep_idx = get_shown_metrics(methods_dict)
    if not keep_idx:
        ax.text(0.5, 0.5, "No comparable metrics", ha="center", va="center",
                 transform=ax.transAxes, fontsize=18 * s, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=26 * s, loc="left", fontweight="bold")
        return

    metric_labels = [METRIC_NAMES[i] for i in keep_idx]
    method_names = list(methods_dict.keys())
    n_methods = len(method_names)
    n_metrics = len(keep_idx)
    x = np.arange(n_metrics)
    width = 0.85 / n_methods

    for i, name in enumerate(method_names):
        offset = (i - (n_methods - 1) / 2) * width
        color = GLOBAL_COLOR_MAP[name]
        hatch = "//" if is_multi_task(name) else None

        plot_x, plot_v = [], []
        for xi, j in zip(x, keep_idx):
            v = methods_dict[name][j]
            if v is not None:
                plot_x.append(xi + offset)
                plot_v.append(v)
        ax.bar(plot_x, plot_v, width, label=name, color=color, hatch=hatch,
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=22 * s)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=26 * s, loc="left", fontweight="bold")
    ax.legend(fontsize=18 * s, frameon=False, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), ncol=1,
              handlelength=1.6, handletextpad=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=18 * s)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)


def draw_radar_panel(ax, methods_dict, title, fontsize_scale=1.0):
    """See draw_bar_panel's fontsize_scale docstring."""
    s = fontsize_scale
    methods_dict = reorder_methods(methods_dict)
    keep_idx = get_shown_metrics(methods_dict, min_methods=len(methods_dict))

    metrics = [METRIC_NAMES[i] for i in keep_idx]
    n_axes = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    prime_name = "PRIME-ISP"
    for name, values in methods_dict.items():
        vals = [values[i] for i in keep_idx]
        vals += vals[:1]
        color = GLOBAL_COLOR_MAP[name]

        if name == prime_name:
            ax.plot(angles, vals, linewidth=3.6 * s, color=color, label=name,
                     marker="o", markersize=6 * s, zorder=10)
        elif is_multi_task(name):
            ax.plot(angles, vals, linewidth=2.8 * s, color=color,
                     label=name, marker="o", markersize=5 * s, linestyle="-", zorder=8)
        else:
            ax.plot(angles, vals, linewidth=2.2 * s, color=color,
                     label=name, linestyle=(0, (4, 1.5)), alpha=1.0, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=22 * s)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=15 * s, color="gray")
    ax.set_title(title, fontsize=26 * s, pad=26, fontweight="bold")
    ax.legend(fontsize=20 * s, frameon=False, loc="upper left",
              bbox_to_anchor=(1.15, 1.1), ncol=1,
              handlelength=2.4 * s, handletextpad=0.8 * s,
              labelspacing=0.8 * s, borderaxespad=0.6 * s)
    ax.grid(alpha=0.3, linewidth=1.0 * s)


def draw_image_panel(ax, img_path, label):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if img_path and os.path.exists(img_path):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(label, fontsize=22, fontweight="bold")
    else:
        ax.text(0.5, 0.5, f"{label}\n(not yet available)",
                 ha="center", va="center", fontsize=18, color="gray",
                 transform=ax.transAxes)
        ax.set_facecolor("#F5F5F5")


def draw_placeholder_slot(fig, gs, grid_row, col0, task_name):
    """Renders a single 'data not yet available' panel spanning all 3
    sub-columns of a slot, used when a task isn't in TASKS yet (so ROWS can
    be locked in before every result is collected)."""
    ax = fig.add_subplot(gs[grid_row, col0:col0 + 3])
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#FAFAFA")
    ax.text(0.5, 0.5, f"{task_name}\n(results not yet available)",
             ha="center", va="center", fontsize=24, color="gray",
             transform=ax.transAxes)


TP_TN_FP_FN_COLORS = {
    "TP": "#E8837A",
    "TN": "#D9D9D9",
    "FP": "#7FB8D9",
    "FN": "#8E3FA8",
}


def draw_slot(fig, gs, grid_row, col0, task_name, tasks_dict):
    """Draws one slot (chart + front + back + F1/MCC caption) at grid
    columns [col0, col0+1, col0+2]. Falls back to a placeholder panel if the
    task has no methods/images yet."""
    task_data = tasks_dict.get(task_name)
    methods_dict = task_data["methods"] if task_data else {}

    if not task_data or not methods_dict:
        draw_placeholder_slot(fig, gs, grid_row, col0, task_name)
        return

    n_common = len(get_shown_metrics(methods_dict, min_methods=len(methods_dict)))

    if n_common >= RADAR_MIN_METRICS:
        ax_main = fig.add_subplot(gs[grid_row, col0], projection="polar")
        draw_radar_panel(ax_main, methods_dict, task_name)
    else:
        ax_main = fig.add_subplot(gs[grid_row, col0])
        draw_bar_panel(ax_main, methods_dict, task_name)

    ax_front = fig.add_subplot(gs[grid_row, col0 + 1])
    draw_image_panel(ax_front, task_data.get("best_case_front_img"), "Front")

    ax_back = fig.add_subplot(gs[grid_row, col0 + 2])
    draw_image_panel(ax_back, task_data.get("best_case_back_img"), "Back")

    prime_vals = methods_dict.get("PRIME-ISP")
    if prime_vals is not None:
        f1, mcc = prime_vals[0], prime_vals[1]
        caption = f"F1 = {f1:.3f}    MCC = {mcc:.3f}"
        fig.canvas.draw()  # ensure axes positions are finalized
        x_center = (ax_front.get_position().x0 + ax_back.get_position().x1) / 2
        y_bottom = min(ax_front.get_position().y0, ax_back.get_position().y0) - 0.012
        fig.text(x_center, y_bottom, caption, ha="center", va="top", fontsize=20, fontweight="bold")


def make_figure(rows, tasks_dict, output_path):
    n_rows = len(rows)
    n_cols = 6  # 2 slots x (chart, front, back)
    fig = plt.figure(figsize=(40, 0.7 + 6.2 * n_rows))

    gs = gridspec.GridSpec(
        n_rows + 1, n_cols,
        height_ratios=[0.45] + [1] * n_rows,
        width_ratios=[2.8, 1, 1, 2.8, 1, 1],
        wspace=0.35, hspace=0.4,
    )

    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    header_ax.text(0.5, 0.85, "Benchmark Results Across Interaction Types",
                    transform=header_ax.transAxes, ha="center", va="center",
                    fontsize=30, fontweight="bold")
    header_ax.text(0.5, 0.55,
                    "Each panel: quantitative comparison (left) and qualitative, front / back (right)",
                    transform=header_ax.transAxes, ha="center", va="center",
                    fontsize=19, style="italic", color="dimgray")

    legend_patches = [mpatches.Patch(facecolor=color, label=label)
                       for label, color in TP_TN_FP_FN_COLORS.items()]
    header_ax.legend(handles=legend_patches, loc="center",
                       bbox_to_anchor=(0.5, 0.15), ncol=4, frameon=False,
                       fontsize=22, title="Residue classification", title_fontsize=22)

    for row_idx, (slot1_name, slot2_name) in enumerate(rows):
        grid_row = row_idx + 1
        draw_slot(fig, gs, grid_row, 0, slot1_name, tasks_dict)
        draw_slot(fig, gs, grid_row, 3, slot2_name, tasks_dict)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {output_path}")

    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    plt.close(fig)


if __name__ == "__main__":
    GLOBAL_COLOR_MAP = build_global_color_map(TASKS)
    os.makedirs("./visualizations", exist_ok=True)
    make_figure(ROWS, TASKS, "./visualizations/full_results_figure.png")