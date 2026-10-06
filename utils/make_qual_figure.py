"""
make_qual_figure.py

Qualitative-only companion to make_full_results_figure.py: one representative
task per interaction class (7 classes -> 7 front/back structure pairs),
instead of all 14 test sets, since showing every test set's structure was
the main source of clutter in the combined figure. Within each class with
multiple test sets, the representative is the one with the highest
PRIME-ISP F1 score among sets that currently have data.

Reuses TASKS from make_full_results_figure.py so this stays in sync with a
single source of data.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import os

from make_full_results_figure import TASKS, TP_TN_FP_FN_COLORS

QUAL_REPRESENTATIVES = [
    ("PPI",    "PPI-60"),
    ("PepPI",  "PepPI-639"),
    ("DPI",    "DPI-129 (DNA)"),
    ("RPI",    "RPI (RNA)"),
    ("CarbPI", "CarbPI"),
    ("SMPI",   "SMPI"),
    ("MPI",    "MPI-Ca (Metal)"),
    ("MPI",    "MPI-Fe (Metal)"),
    ("MPI",    "MPI-Mn (Metal)"),
]

QUAL_ROWS = [
    QUAL_REPRESENTATIVES[0:3],
    QUAL_REPRESENTATIVES[3:6],
    QUAL_REPRESENTATIVES[6:9],
]


def draw_qual_panel(fig, gs, row, col0, class_name, task_name, tasks_dict):
    task_data = tasks_dict.get(task_name, {})

    ax_front = fig.add_subplot(gs[row, col0])
    ax_back = fig.add_subplot(gs[row, col0 + 1])

    for ax, key, label in [(ax_front, "best_case_front_img", "Front"),
                            (ax_back, "best_case_back_img", "Back")]:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        img_path = task_data.get(key)
        if img_path and os.path.exists(img_path):
            ax.imshow(plt.imread(img_path))
        else:
            ax.text(0.5, 0.5, "(not yet\navailable)", ha="center", va="center",
                     fontsize=14, color="gray", transform=ax.transAxes)
            ax.set_facecolor("#F5F5F5")
        ax.set_title(label, fontsize=15, fontweight="bold")

    fig.canvas.draw()
    x_center = (ax_front.get_position().x0 + ax_back.get_position().x1) / 2
    y_top = max(ax_front.get_position().y1, ax_back.get_position().y1)
    fig.text(x_center, y_top + 0.012, task_name.split(" (")[0],
              ha="center", va="bottom", fontsize=18, fontweight="bold")

    # Prefer this specific rendered case's OWN F1/MCC (exported by
    # render_prime_predictions.py as "<prefix>_metrics.json", alongside
    # "<prefix>_front.png") over the whole-test-set aggregate PRIME-ISP
    # score in TASKS -- the pictured protein is a hand-picked best/worst
    # case, not the dataset average, so captioning it with the aggregate
    # number would describe a different protein than the one on screen.
    f1 = mcc = None
    is_per_case = False
    front_img = task_data.get("best_case_front_img")
    if front_img and front_img.endswith("_front.png"):
        metrics_path = front_img[: -len("_front.png")] + "_metrics.json"
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    case_metrics = json.load(f)
                f1, mcc = case_metrics["f1"], case_metrics["mcc"]
                is_per_case = True
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Warning: couldn't read {metrics_path} ({e}); "
                      f"falling back to the aggregate score for {task_name}.")

    if f1 is None:  # fall back to the aggregate PRIME-ISP score from TASKS
        methods_dict = task_data.get("methods", {})
        prime_vals = methods_dict.get("PRIME-ISP ")
        if prime_vals is not None:
            f1, mcc = prime_vals[0], prime_vals[1]

    if f1 is not None:
        caption = f"F1 = {f1:.3f}    MCC = {mcc:.3f}"
        if not is_per_case:
            caption += "  (test-set avg.)"
        y_bottom = min(ax_front.get_position().y0, ax_back.get_position().y0) - 0.014
        fig.text(x_center, y_bottom, caption, ha="center", va="top",
                  fontsize=15, fontweight="bold")


def make_figure(rows, tasks_dict, output_path):
    n_rows = len(rows)
    n_cols_per_slot = 2  # front, back
    max_slots = max(len(r) for r in rows)
    n_cols = max_slots * n_cols_per_slot
    fig = plt.figure(figsize=(4.6 * n_cols, 0.7 + 5.6 * n_rows))

    gs = gridspec.GridSpec(
        n_rows + 1, n_cols,
        height_ratios=[0.45] + [1] * n_rows,
        wspace=0.15, hspace=0.55,
    )

    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    header_ax.text(0.5, 0.6,
                    "Qualitative Comparison on Residue-Level Interaction-Site Prediction",
                    transform=header_ax.transAxes, ha="center", va="center",
                    fontsize=26, fontweight="bold")
    header_ax.text(0.5, 0.3,
                    "One representative test set per interaction class",
                    transform=header_ax.transAxes, ha="center", va="center",
                    fontsize=16, style="italic", color="dimgray")

    legend_patches = [mpatches.Patch(facecolor=color, label=label)
                       for label, color in TP_TN_FP_FN_COLORS.items()]
    header_ax.legend(handles=legend_patches, loc="center",
                       bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False,
                       fontsize=18, title="Residue classification", title_fontsize=18)

    for row_idx, slots in enumerate(rows):
        grid_row = row_idx + 1
        col_offset = (n_cols - len(slots) * n_cols_per_slot) // 2
        for i, (class_name, task_name) in enumerate(slots):
            col0 = col_offset + i * n_cols_per_slot
            draw_qual_panel(fig, gs, grid_row, col0, class_name, task_name, tasks_dict)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    os.makedirs("./visualizations", exist_ok=True)
    make_figure(QUAL_ROWS, TASKS, "./visualizations/qualitative_results_figure.png")