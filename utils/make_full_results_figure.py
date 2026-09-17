"""
make_full_results_figure.py

One row per task, 3 panels: (1) radar chart if >=4 metrics are shared across
methods, otherwise a bar chart; (2) best-case structure render; (3) worst-case
structure render.

No "(SOTA)" labeling -- every method is presented on equal footing. Multi-task
/ unified frameworks (ALLSites, PRIME-ISP) are visually distinguished from
single-task specialized methods: hatched bars in the bar-chart panels, solid
+ marker lines (vs. thin dashed lines) in the radar panels.

Tasks split into 3 groups by interaction category, each its own figure file.
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
MULTI_TASK_METHODS = {"ALLSites", "PRIME-ISP (ours)"}

TASKS = {
    "RPI (RNA)": {
        "methods": {
            "RNABindRPlus": [0.248, 0.202, 0.227, 0.273, 0.717],
            "VMnuc":        [0.235, 0.192, 0.240, 0.231, 0.729],
            "COACH-D":      [0.235, 0.195, 0.252, 0.221, 0.663],
            "NucBind":      [0.233, 0.189, 0.235, 0.231, 0.715],
            "aaRNA":        [0.247, 0.214, 0.166, 0.484, 0.771],
            "NucleicNet":   [0.261, 0.216, 0.201, 0.371, 0.788],
            "GraphPBSP":    [0.298, 0.275, 0.264, 0.512, 0.795],
            "GraphBind":    [0.358, 0.322, 0.294, 0.463, 0.854],
            "ALLSites":     [0.336, 0.303, 0.277, 0.457, 0.853],
            "DeepPANB":     [0.391, 0.354, 0.351, 0.481, 0.871],
            "PRIME-ISP (ours)":[0.374, 0.346, 0.286, 0.541, 0.871],
        },
        "best_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
        "worst_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
    },
    "DPI-129 (DNA)": {
        "methods": {
            "COACH-D":      [0.323, 0.279, 0.318, 0.328, 0.712],
            "NucBind":      [0.343, 0.304, 0.366, 0.322, 0.809],
            "GraphSite":    [0.441, 0.425, 0.423, 0.566, 0.912],
            "GLMSite":      [0.405, 0.412, 0.287, 0.848, 0.918],
            "GraphBind":    [0.522, 0.499, 0.425, 0.676, 0.927],
            "EquiPNAS":     [0.462, 0.443, 0.471, 0.516, 0.919],
            "EGPDI":        [0.549, 0.522, 0.503, 0.612, 0.941],
            "ALLSites":     [0.501, 0.480, 0.401, 0.674, 0.927],
            "DeepPANB":     [0.592, 0.564, 0.559, 0.629, 0.945],
            "PRIME-ISP (ours)":[0.548, 0.533, 0.435, 0.740, 0.941],
        },
        "best_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
        "worst_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
    },
    "DPI-181 (DNA)": {
        "methods": {
            "COACH-D":      [0.251, 0.220, 0.266, 0.239, 0.668],
            "NucBind":      [0.262, 0.227, 0.240, 0.288, 0.798],
            "GraphSite":    [0.345, 0.332, 0.343, 0.454, 0.892],
            "GLMSite":      [0.311, 0.334, 0.209, 0.829, 0.899],
            "GraphBind":    [0.379, 0.370, 0.331, 0.566, 0.895],
            "EquiPNAS":     [0.366, 0.353, 0.346, 0.436, 0.907],
            "EGPDI":        [0.424, 0.407, 0.346, 0.558, 0.914],
            "ALLSites":     [0.389, 0.379, 0.292, 0.595, 0.904],
            "DeepPANB":     [0.435, 0.419, 0.407, 0.560, 0.920],
            "PRIME-ISP (ours)":[0.402, 0.402, 0.287, 0.667, 0.917],
        },
        "best_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
        "worst_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
    },
    "PepPI": {
        "methods": {
            "PepBind":           [None,  0.348, None, None, 0.767],
            "PepNN-Struct":      [None,  0.301, None, None, 0.838],
            "ALLSites":          [None,  0.316, None, None, 0.817],
            "DeepProSite":       [None,  0.397, None, None, 0.861],
            "PRIME-ISP (ours)":  [0.376, 0.345, 0.290, 0.536, 0.850],
        },
        "best_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
        "worst_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
    },
    "PPI-60": {
        "methods": {
            "RGN":               [0.455, 0.344, None, None, 0.779],
            "ALLSites":          [0.448, 0.340, None, None, 0.781],
            "DeepProSite":       [0.470, 0.379, None, None, 0.813],
            "PRIME-ISP (ours)":  [0.466, 0.353, 0.397, 0.564, 0.797],
        },
        "best_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
        "worst_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
    },
    "PPI-315": {
        "methods": {
            "RGN":               [0.390, 0.273, None, None, 0.752],
            "ALLSites":          [0.392, 0.273, None, None, 0.746],
            "DeepProSite":       [0.457, 0.355, None, None, 0.805],
            "PRIME-ISP (ours)":  [0.431, 0.323, 0.339, 0.594, 0.789],
        },
        "best_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
        "worst_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
    },
    "MPI-Ca (Metal)": {
        "methods": {
            "MetalBind":       [0.506, 0.527, 0.720, 0.390, 0.952],
            "PRIME-ISP (ours)":[0.671, 0.670, 0.718, 0.629, 0.950],
        },
        "best_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
        "worst_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
    },
    "MPI-Fe (Metal)": {
        "methods": {
            "MetalBind":       [0.676, 0.674, 0.674, 0.679, 0.981],
            "PRIME-ISP (ours)":[0.678, 0.680, 0.611, 0.763, 0.963],
        },
        "best_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
        "worst_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
    },
    "MPI-Mn (Metal)": {
        "methods": {
            "MetalBind":       [0.518, 0.552, 0.615, 0.447, 0.953],
            "PRIME-ISP (ours)":[0.559, 0.558, 0.604, 0.521, 0.948],
        },
        "best_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
        "worst_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
    },
    "CarbPI": {
        "methods": {
            "FTMap":           [0.351, 0.222, 0.284, 0.505, None],
            "CAPSIF:G":        [0.543, 0.538, 0.541, 0.590, None],
            "CAPSIF:V":        [0.597, 0.599, 0.598, 0.647, None],
            "ALLSites":        [0.609, 0.603, 0.624, 0.652, None],
            "PRIME-ISP (ours)":[0.665, 0.655, 0.593, 0.757, 0.961],
        },
        "best_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
        "worst_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
    },
    "SMPI": {
        "methods": {
            "P2Rank":          [0.450, 0.424, 0.597, 0.361, None],
            "ALLSites":        [0.601, 0.560, 0.609, 0.593, 0.926],
            "PRIME-ISP (ours)":[0.601, 0.560, 0.609, 0.593, 0.926],
        },
        "best_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
        "worst_case_img": "/home/dvnguye2/PRIME_ISP/visualizations/DPI-site/5f55_A_back.png",
    },
}

GROUPS = {
    "figure_A_nucleic_acid": ["RPI (RNA)", "DPI-129 (DNA)", "DPI-181 (DNA)"],
    "figure_B_protein":      ["PepPI", "PPI-60", "PPI-315"],
    "figure_C_metal_carb_small": ["MPI-Ca (Metal)", "MPI-Fe (Metal)", "MPI-Mn (Metal)",
                                   "CarbPI", "SMPI"],
}
 
COLOR_PRIME = "#C0392B"
COLOR_PALETTE = plt.get_cmap("tab20").colors
RADAR_MIN_METRICS = 4
 
 
def build_global_color_map(tasks_dict):
    """Assign each method name a fixed color, consistent across every task
    it appears in -- so e.g. ALLSites or GraphBind always look the same
    regardless of which figure/panel they're shown in."""
    all_names = set()
    for task_data in tasks_dict.values():
        all_names.update(task_data["methods"].keys())
    all_names.discard("PRIME-ISP (ours)")  # always red, handled separately
 
    color_map = {"PRIME-ISP (ours)": COLOR_PRIME}
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
    and PRIME-ISP always sit adjacent in both bars and legend."""
    single_task = {k: v for k, v in methods_dict.items() if not is_multi_task(k)}
    multi_task = {k: v for k, v in methods_dict.items() if is_multi_task(k)}
    # within multi-task group, keep ALLSites before PRIME-ISP if both present
    multi_task_ordered = dict(sorted(multi_task.items(),
                                       key=lambda kv: kv[0] == "PRIME-ISP (ours)"))
    return {**single_task, **multi_task_ordered}
 
 
def draw_bar_panel(ax, methods_dict, title):
    methods_dict = reorder_methods(methods_dict)
    keep_idx = get_shown_metrics(methods_dict)
    if not keep_idx:
        ax.text(0.5, 0.5, "No comparable metrics", ha="center", va="center",
                 transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
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
    ax.set_xticklabels(metric_labels, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="upper left",
              bbox_to_anchor=(1.05, 1.0), ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
 
 
def draw_radar_panel(ax, methods_dict, title):
    methods_dict = reorder_methods(methods_dict)
    keep_idx = get_shown_metrics(methods_dict, min_methods=len(methods_dict))
    # radar requires ALL methods share the axis, unlike bar's "at least 2"
    if len(keep_idx) < RADAR_MIN_METRICS:
        keep_idx = get_shown_metrics(methods_dict, min_methods=len(methods_dict))
 
    metrics = [METRIC_NAMES[i] for i in keep_idx]
    n_axes = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]
 
    for name, values in methods_dict.items():
        vals = [values[i] for i in keep_idx]
        vals += vals[:1]
        color = GLOBAL_COLOR_MAP[name]
 
        if name == "PRIME-ISP (ours)":
            ax.plot(angles, vals, linewidth=2.2, color=color, label=name,
                     marker="o", markersize=3, zorder=10)
            ax.fill(angles, vals, alpha=0.12, color=color, zorder=10)
        elif is_multi_task(name):
            ax.plot(angles, vals, linewidth=1.6, color=color,
                     label=name, marker="o", markersize=3, linestyle="-")
        else:
            ax.plot(angles, vals, linewidth=0.9, color=color,
                     label=name, linestyle="--", alpha=0.75)
 
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6, color="gray")
    ax.set_title(title, fontsize=10, pad=14)
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.15, 1.1))
    ax.grid(alpha=0.3)
 
 
def draw_image_panel(ax, img_path, label):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
 
    if img_path and os.path.exists(img_path):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(label, fontsize=9)
    else:
        ax.text(0.5, 0.5, f"{label}\n(not yet available)",
                 ha="center", va="center", fontsize=8, color="gray",
                 transform=ax.transAxes)
        ax.set_facecolor("#F5F5F5")
 
 
TP_TN_FP_FN_COLORS = {
    "TP": "#E8837A",
    "TN": "#D9D9D9",
    "FP": "#7FB8D9",
    "FN": "#8E3FA8",
}
 
 
def make_figure(task_names, tasks_dict, output_path):
    task_names = [t for t in task_names if t in tasks_dict]
    n_tasks = len(task_names)
    fig = plt.figure(figsize=(17, 2.9 * n_tasks))
    gs = gridspec.GridSpec(n_tasks, 3, width_ratios=[1.8, 1, 1], wspace=0.35, hspace=0.5)
 
    for row, task_name in enumerate(task_names):
        task_data = tasks_dict[task_name]
        n_common = len(get_shown_metrics(task_data["methods"], min_methods=len(task_data["methods"])))
 
        if n_common >= RADAR_MIN_METRICS:
            ax_main = fig.add_subplot(gs[row, 0], projection="polar")
            draw_radar_panel(ax_main, task_data["methods"], task_name)
        else:
            ax_main = fig.add_subplot(gs[row, 0])
            draw_bar_panel(ax_main, task_data["methods"], task_name)
 
        ax_best = fig.add_subplot(gs[row, 1])
        draw_image_panel(ax_best, task_data.get("best_case_img"), "Best case")
 
        ax_worst = fig.add_subplot(gs[row, 2])
        draw_image_panel(ax_worst, task_data.get("worst_case_img"), "Worst case")
 
    # One shared TP/TN/FP/FN legend for all structure-render panels in this
    # figure, instead of repeating it per row.
    legend_patches = [mpatches.Patch(facecolor=color, label=label)
                       for label, color in TP_TN_FP_FN_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower right",
               bbox_to_anchor=(0.98, 1.005), ncol=4, frameon=False, fontsize=10,
               title="Residue classification", title_fontsize=10)
 
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    GLOBAL_COLOR_MAP = build_global_color_map(TASKS)
    for fig_name, task_names in GROUPS.items():
        make_figure(task_names, TASKS, f"./visualizations/{fig_name}.png")