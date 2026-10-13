import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ==================== COLOR SCHEME ====================
COLORS = {
    "basemodel": ("#4C72B0", "#DCE6F1"),
    "cva":       ("#8172B2", "#E9E1F3"),
    "process":   ("#DD8452", "#FCE9DC"),
    "final":     ("#55A868", "#DFF0E3"),
}
FADE_ALPHA = 0.18

# ==================== LAYOUT CONSTANTS ====================
COL_SPACING = 2.3          # distance between adjacent column centers
NODE_W, NODE_H = 2.0, 1.15  # process/base/cva box size (wide enough for 2 lines at size 14/12)
FINAL_W, FINAL_H = 2.0, 0.9  # final-model boxes only need one line

TITLE_SIZE, SUBTITLE_SIZE = 16, 14

# Column centers (5 columns)
col_x = [i * COL_SPACING for i in range(5)]  # 0, 2.3, 4.6, 6.9, 9.2

ROW_FINAL = -0.8
ROW_PROCESS = 1.5
ROW_CVA = 2.95
ROW_BASE = 5.1

# ==================== NODE DEFINITIONS ====================
# key: (center_x, center_y, width, height, title, subtitle_or_None, category)
nodes = {
    "gpt_en": (col_x[0], ROW_BASE, NODE_W, NODE_H, "GPT-2 English", "(124M / OpenAI)", "basemodel"),
    "gpt_es": ((col_x[1] + col_x[4]) / 2, ROW_BASE, NODE_W, NODE_H, "GPT-2 Spanish", "(117M / DeepESP)", "basemodel"),

    "pipe_cscl_en":  (col_x[0], ROW_PROCESS, NODE_W, NODE_H, "3-Phase CSCL", "(English Curriculum)", "process"),
    "pipe_mono":     (col_x[1], ROW_PROCESS, NODE_W, NODE_H, "Monolingual\n Baseline", "Basque Pre-training", "process"),
    "pipe_cscl_es":  (col_x[2], ROW_PROCESS, NODE_W, NODE_H, "3-Phase CSCL", "(Spanish Curriculum)", "process"),
    "pipe_tok_cscl": (col_x[3], ROW_PROCESS, NODE_W, NODE_H, "3-Phase CSCL", "Pipeline", "process"),
    "pipe_ener_cscl":(col_x[4], ROW_PROCESS, NODE_W, NODE_H, "3-Phase CSCL", "Pipeline", "process"),

    "cva_tok":  (col_x[3], ROW_CVA, NODE_W, NODE_H, "TokAlign", "Stabilized", "cva"),
    "cva_ener": (col_x[4], ROW_CVA, NODE_W, NODE_H, "EnerGIZAr", "Untrained Init", "cva"),

    "m_cscl_en": (col_x[0], ROW_FINAL, FINAL_W, FINAL_H, "CSCL-en", None, "final"),
    "m_base":    (col_x[1], ROW_FINAL, FINAL_W, FINAL_H, "Baseline", None, "final"),
    "m_cscl_es": (col_x[2], ROW_FINAL, FINAL_W, FINAL_H, "CSCL-es", None, "final"),
    "m_tok":     (col_x[3], ROW_FINAL, FINAL_W, FINAL_H, "TokAlign-CSCL", None, "final"),
    "m_ener":    (col_x[4], ROW_FINAL, FINAL_W, FINAL_H, "EnerGIZAr-CSCL", None, "final"),
}

arrows = [
    ("gpt_en", "pipe_cscl_en", "straight"),
    ("pipe_cscl_en", "m_cscl_en", "straight"),

    ("gpt_es", "pipe_mono", "elbow"),
    ("gpt_es", "pipe_cscl_es", "elbow"),
    ("gpt_es", "cva_tok", "elbow"),
    ("gpt_es", "cva_ener", "elbow"),

    ("pipe_mono", "m_base", "straight"),
    ("pipe_cscl_es", "m_cscl_es", "straight"),

    ("cva_tok", "pipe_tok_cscl", "straight"),
    ("pipe_tok_cscl", "m_tok", "straight"),

    ("cva_ener", "pipe_ener_cscl", "straight"),
    ("pipe_ener_cscl", "m_ener", "straight"),
]


def node_bottom(key):
    cx, cy, w, h, *_ = nodes[key]
    return (cx, cy - h / 2)


def node_top(key):
    cx, cy, w, h, *_ = nodes[key]
    return (cx, cy + h / 2)


def draw_box(ax, key, active):
    cx, cy, w, h, title, subtitle, category = nodes[key]
    edge, fill = COLORS[category]
    alpha = 1.0 if active else FADE_ALPHA
    text_alpha = 1.0 if active else 0.35

    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        linewidth=1.8, edgecolor=edge, facecolor=fill,
        alpha=alpha, zorder=3,
    )
    ax.add_patch(box)

    if subtitle:
        ax.text(cx, cy + h * 0.16, title, ha="center", va="center",
                 fontsize=TITLE_SIZE, fontweight="bold", zorder=4, alpha=text_alpha)
        ax.text(cx, cy - h * 0.22, subtitle, ha="center", va="center",
                 fontsize=SUBTITLE_SIZE, zorder=4, alpha=text_alpha)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                 fontsize=TITLE_SIZE, fontweight="bold", zorder=4, alpha=text_alpha)


def draw_arrow(ax, from_key, to_key, style, active):
    color = "#606060" if active else "#d0d0d0"
    alpha = 0.95 if active else 0.5
    lw = 1.6 if active else 0.9

    start = node_bottom(from_key)
    end = node_top(to_key)

    if style == "straight":
        a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                              linewidth=lw, color=color, alpha=alpha, zorder=2)
        ax.add_patch(a)
    else:
        mid_y = start[1] - 0.35
        ax.plot([start[0], start[0], end[0]], [start[1], mid_y, mid_y],
                 color=color, linewidth=lw, alpha=alpha, zorder=2,
                 solid_capstyle="round")
        a = FancyArrowPatch((end[0], mid_y), end, arrowstyle="-|>", mutation_scale=13,
                              linewidth=lw, color=color, alpha=alpha, zorder=2)
        ax.add_patch(a)


def plot_taxonomy(highlight=None, title=None, output_path="plots/taxonomy.pdf"):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.set_xlim(col_x[0] - NODE_W / 2 - 0.4, col_x[4] + NODE_W / 2 + 0.4)
    ax.set_ylim(ROW_FINAL - FINAL_H / 2 - 0.4, ROW_BASE + NODE_H / 2 + 0.7)

    def is_active(key):
        return highlight is None or key in highlight

    for from_key, to_key, style in arrows:
        active = is_active(from_key) and is_active(to_key)
        draw_arrow(ax, from_key, to_key, style, active)

    for key in nodes:
        draw_box(ax, key, is_active(key))

    legend_handles = [
        mpatches.Patch(facecolor=COLORS["basemodel"][1], edgecolor=COLORS["basemodel"][0], label="Base Models"),
        mpatches.Patch(facecolor=COLORS["cva"][1], edgecolor=COLORS["cva"][0], label="CVA checkpoints"),
        mpatches.Patch(facecolor=COLORS["process"][1], edgecolor=COLORS["process"][0], label="Pre-training Configuration"),
        mpatches.Patch(facecolor=COLORS["final"][1], edgecolor=COLORS["final"][0], label="Final Models"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=4, frameon=False, fontsize=14)

    if title:
        ax.text((col_x[0] + col_x[4]) / 2, ROW_FINAL - FINAL_H / 2 - 0.3, title,
                 ha="center", va="top", fontsize=14, style="italic")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    plot_taxonomy(highlight=None, output_path="plots/taxonomy_full.pdf")
    plot_taxonomy(highlight={"gpt_en", "gpt_es"}, output_path="plots/taxonomy_basemodels.pdf")
    plot_taxonomy(highlight={"gpt_es", "pipe_mono", "m_base"}, output_path="plots/taxonomy_cpt.pdf")
    plot_taxonomy(highlight={"gpt_en", "gpt_es", "pipe_cscl_en", "pipe_cscl_es", "m_cscl_en", "m_cscl_es"},
                   output_path="plots/taxonomy_cscl.pdf")
    plot_taxonomy(highlight={"gpt_es", "cva_tok", "cva_ener", "pipe_tok_cscl", "pipe_ener_cscl", "m_tok", "m_ener"},
                   output_path="plots/taxonomy_cva.pdf")