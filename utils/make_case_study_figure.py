"""
make_case_study_figure.py

Composite qualitative figure for the "Case Studies: Partner-Specific
Prediction on Multi-Site Proteins" subsection.

Layout: one row per case-study protein (thrombin, ER-alpha, calmodulin);
each row has one slot per partner-specific site on that same structure
(2 sites x front/back = 4 image panels per row), captioned with that
site's own P/R/F1/MCC (read from the per-record "<record_id>_metrics.json"
that render_case_study_with_partner.py exports alongside its PNGs -- these
are the actual metrics for the pictured structure, not a test-set average,
since case-study targets aren't part of any task's test split).

Reuses TP_TN_FP_FN_COLORS from make_full_results_figure.py so the legend
matches every other qualitative figure in the paper.

Usage:
    python utils/make_case_study_figure.py --figs_dir ./Datasets/CaseStudies/figs --output_path ./visualizations/case_study_figure.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from make_full_results_figure import TP_TN_FP_FN_COLORS

# Each row = one case-study protein. Each entry in a row = one partner
# site on that SAME structure: (display_label, record_id). record_id must
# match what run_case_study_inference.py / render_case_study_with_partner.py
# used, i.e. "{target}__{site_name}". Labels are just the interaction-class
# task name -- the partner's identity is in the LaTeX prose/caption, not
# repeated on the panel itself.
CASE_STUDY_ROWS = [
    ("Thrombin (8TQS)", [
        ("DPI", "thrombin__DPI_exosite_II"),
        ("SMPI", "thrombin__SMPI_active_site"),
    ]),
    (r"ER$\alpha$ (3L03)", [
        ("SMPI", "er_alpha__SMPI_orthosteric"),
        ("PepPI", "er_alpha__PepPI_coactivator"),
    ]),
    ("Calmodulin (1CDL)", [
        ("MPI", "calmodulin__MPI_calcium"),
        ("PepPI", "calmodulin__PepPI_target"),
    ]),
]


def draw_site_panel(fig, gs, row, col0, label, record_id, figs_dir):
    ax_front = fig.add_subplot(gs[row, col0])
    ax_back = fig.add_subplot(gs[row, col0 + 1])

    for ax, suffix, view_label in [(ax_front, "_front.png", "Front"),
                                     (ax_back, "_back.png", "Back")]:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        img_path = os.path.join(figs_dir, f"{record_id}{suffix}")
        if os.path.exists(img_path):
            ax.imshow(plt.imread(img_path))
        else:
            ax.text(0.5, 0.5, "(not yet\navailable)", ha="center", va="center",
                     fontsize=13, color="gray", transform=ax.transAxes)
            ax.set_facecolor("#F5F5F5")
        ax.set_title(view_label, fontsize=13, fontweight="bold")

    fig.canvas.draw()
    x_center = (ax_front.get_position().x0 + ax_back.get_position().x1) / 2
    y_top = max(ax_front.get_position().y1, ax_back.get_position().y1)
    fig.text(x_center, y_top + 0.015, label,
              ha="center", va="bottom", fontsize=14, fontweight="bold")

    metrics_path = os.path.join(figs_dir, f"{record_id}_metrics.json")
    y_bottom = min(ax_front.get_position().y0, ax_back.get_position().y0) - 0.012
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        caption = (f"Precision={m['precision']:.3f}  Recall={m['recall']:.3f}  "
                   f"F1={m['f1']:.3f}  MCC={m['mcc']:.3f}")
        fig.text(x_center, y_bottom, caption, ha="center", va="top",
                  fontsize=12.5, fontweight="bold")
    else:
        fig.text(x_center, y_bottom, "(metrics not found)", ha="center", va="top",
                  fontsize=12.5, color="gray", style="italic")


def draw_row_label(fig, gs, row, protein_label):
    """Left-side row label spanning the row's height."""
    label_ax = fig.add_subplot(gs[row, 0])
    label_ax.axis("off")
    fig.canvas.draw()
    pos = label_ax.get_position()
    fig.text(pos.x0 - 0.01, (pos.y0 + pos.y1) / 2, protein_label,
              ha="right", va="center", fontsize=17, fontweight="bold", rotation=90)


def make_figure(rows, figs_dir, output_path):
    n_rows = len(rows)
    n_slots = max(len(r[1]) for r in rows)
    n_cols_per_slot = 2  # front, back
    n_cols = 1 + n_slots * n_cols_per_slot  # +1 narrow col for the row label

    fig = plt.figure(figsize=(4.6 * n_slots * n_cols_per_slot + 1.0, 0.9 + 5.6 * n_rows))
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols,
        width_ratios=[0.18] + [1] * (n_cols - 1),
        height_ratios=[0.4] + [1] * n_rows,
        wspace=0.15, hspace=0.6,
    )

    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    header_ax.text(0.5, 0.6,
                    "Case Studies: Partner-Specific Prediction on Multi-Site Proteins",
                    transform=header_ax.transAxes, ha="center", va="center",
                    fontsize=24, fontweight="bold")
    header_ax.text(0.5, 0.25,
                    "Each row: one co-crystal structure; each partner class "
                    "scored independently from the same protein input.",
                    transform=header_ax.transAxes, ha="center", va="center",
                    fontsize=14, style="italic", color="dimgray")

    legend_patches = [mpatches.Patch(facecolor=color, label=lab)
                       for lab, color in TP_TN_FP_FN_COLORS.items()]
    # PyMOL's "forest" ~ a standard dark green; matches PARTNER_COLOR in
    # render_case_study_with_partner.py. Kept as a literal here rather than
    # imported since it's a PyMOL color NAME, not a hex/rgb value like the
    # TP/TN/FP/FN colors are.
    # legend_patches.append(mpatches.Patch(facecolor="#228B22", label="Partner (bound molecule)"))
    header_ax.legend(handles=legend_patches, loc="center",
                      bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False,
                      fontsize=16, title="Residue classification", title_fontsize=16)

    for row_idx, (protein_label, sites) in enumerate(rows):
        grid_row = row_idx + 1
        draw_row_label(fig, gs, grid_row, protein_label)
        n_this_row = len(sites)
        col_offset = 1 + ((n_slots - n_this_row) * n_cols_per_slot) // 2
        for i, (label, record_id) in enumerate(sites):
            col0 = col_offset + i * n_cols_per_slot
            draw_site_panel(fig, gs, grid_row, col0, label, record_id, figs_dir)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figs_dir", type=str, required=True,
                         help="Directory containing <record_id>_front.png / "
                              "_back.png / _metrics.json from "
                              "render_case_study_with_partner.py.")
    parser.add_argument("--output_path", type=str,
                         default="./visualizations/case_study_figure.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    make_figure(CASE_STUDY_ROWS, args.figs_dir, args.output_path)