import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig = plt.figure(figsize=(14, 6.5))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")

BUBBLE_W, BUBBLE_H = 2.4, 0.9

def rounded_box(ax, center, width, height, text, edgecolor, facecolor="white", fontsize=14):
    xy = (center[0] - width / 2, center[1] - height / 2)
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=2, edgecolor=edgecolor, facecolor=facecolor, zorder=3,
    )
    ax.add_patch(box)
    ax.text(center[0], center[1], text, ha="center", va="center",
             fontsize=fontsize, zorder=4)
    return box

def header_above(ax, center, box_height, text, fontsize=16, gap=0.25):
    """Places a bold header ABOVE a box, outside its border -- matching
    how 'Parallel Corpus' sits outside its own bubble."""
    ax.text(center[0], center[1] + box_height / 2 + gap, text,
             ha="center", va="bottom", fontsize=fontsize, fontweight="bold", zorder=4)

def arrow(ax, start, end, color="black"):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                          linewidth=1.6, color=color, zorder=2)
    ax.add_patch(a)

def arrow_label(ax, start, end, text, color="black", offset=(0, 0), fontsize=12, fontweight="bold"):
    mx, my = (start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1]
    ax.text(mx, my, text, ha="center", va="center", fontsize=fontsize,
             fontweight=fontweight, color=color, zorder=5,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

# ---- Parallel corpus bubble ----
corpus_center = (2.2, 3.25)
rounded_box(ax, corpus_center, BUBBLE_W, BUBBLE_H,
            "eu: \"...eskolan, kalean\neta etxean...\"\nes: \"...en la escuela, en\nla calle y en casa...\"",
            edgecolor="black")
ax.text(corpus_center[0] - BUBBLE_W / 2 - 0.2, corpus_center[1], "Parallel\nCorpus",
         ha="right", va="center", fontsize=16, fontweight="bold")

# ---- Tokenizer_s (Spanish) branch, top ----
tok_s_center = (corpus_center[0], corpus_center[1] + 1.9)
rounded_box(ax, tok_s_center, BUBBLE_W, BUBBLE_H,
            "es: 4287, 19, 5502,\n3, 88, 214...",
            edgecolor="#3B9DD6")
arrow(ax, (corpus_center[0], corpus_center[1] + BUBBLE_H / 2),
          (tok_s_center[0], tok_s_center[1] - BUBBLE_H / 2 - 0.1))
ax.text(corpus_center[0] - 0.25, (corpus_center[1] + tok_s_center[1]) / 2,
         r"(1) Tokenizer$_s$", ha="right", va="center", fontsize=16, fontweight="bold")

# ---- Tokenizer_t (Basque) branch, bottom ----
tok_t_center = (corpus_center[0], corpus_center[1] - 1.9)
rounded_box(ax, tok_t_center, BUBBLE_W, BUBBLE_H,
            "eu: 791, 1648, 311,\n636, 3940...",
            edgecolor="#E8A33D")
arrow(ax, (corpus_center[0], corpus_center[1] - BUBBLE_H / 2),
          (tok_t_center[0], tok_t_center[1] + BUBBLE_H / 2 + 0.1))
ax.text(corpus_center[0] - 0.25, (corpus_center[1] + tok_t_center[1]) / 2,
         r"(1) Tokenizer$_t$", ha="right", va="center", fontsize=16, fontweight="bold")

# ---- GIZA++ box: header now OUTSIDE, box shrunk to fit body text only ----
GAP = 1.5
GIZA_W, GIZA_H = 2.4, 1.15
giza_center = (tok_s_center[0] + BUBBLE_W / 2 + GAP + 1.2, corpus_center[1])
rounded_box(ax, giza_center, GIZA_W, GIZA_H,
            "Estimates, for every\nsubword pair, the\ntranslation probability\n$p(y \\mid x)$",
            edgecolor="purple")
header_above(ax, giza_center, GIZA_H, "(2) GIZA++ Alignment")

arrow(ax, (tok_s_center[0] + BUBBLE_W / 2, tok_s_center[1]), (giza_center[0] - GIZA_W / 2 - 0.15, corpus_center[1] + 0.4), color="black")
arrow(ax, (tok_t_center[0] + BUBBLE_W / 2, tok_t_center[1]), (giza_center[0] - GIZA_W / 2 - 0.15, corpus_center[1] - 0.4), color="black")

# ---- D^{t,s} alignment matrix, with "(3) Build Matrix" stacked ABOVE its own title ----
matrix_left = giza_center[0] + GIZA_W / 2 + 1.3
MATRIX_SIZE = 2.0
n = 5
np.random.seed(11)
matrix = np.random.rand(n, n) * 0.4
strong_cells = [(0, 1), (1, 3), (2, 0), (3, 2), (4, 4)]
for (r, c) in strong_cells:
    matrix[r, c] = 0.8

arrow_start = (giza_center[0] + GIZA_W / 2, corpus_center[1])
arrow_end = (matrix_left - 0.15, corpus_center[1])
arrow(ax, arrow_start, arrow_end, color="black")

ax_matrix = ax.inset_axes([matrix_left, corpus_center[1] - MATRIX_SIZE / 2, MATRIX_SIZE, MATRIX_SIZE], transform=ax.transData)
ax_matrix.imshow(matrix, cmap="Purples", vmin=0, vmax=1, aspect="auto")
ax_matrix.set_xticks(range(n)); ax_matrix.set_xticklabels(["0", "1", "2", "...", r"$|V_1|$"], fontsize=14)
ax_matrix.set_yticks(range(n)); ax_matrix.set_yticklabels(["0", "1", "2", "...", r"$|V_2|$"], fontsize=14)
ax_matrix.set_title(r"Alignment Matrix $D^{t,s}$", fontsize=14)
ax.text(matrix_left + MATRIX_SIZE / 2, corpus_center[1] + MATRIX_SIZE / 2 + 0.3,
         "(3) Build Matrix", ha="center", va="bottom", fontsize=16, fontweight="bold", zorder=4)
for spine in ax_matrix.spines.values():
    spine.set_edgecolor("purple"); spine.set_linewidth(2)

# ---- Post-processing box: header OUTSIDE, cleaned-up bullets, shrunk to fit ----
POSTPROC_W, POSTPROC_H = 2.4, 1.2
postproc_center = (matrix_left + MATRIX_SIZE + 1.8, corpus_center[1] + 1.15)
rounded_box(ax, postproc_center, POSTPROC_W, POSTPROC_H,
            "$\\bullet$ zero out entries $< \\delta$\n"
            "$\\bullet$ fall back to random init\n"
            "$\\bullet$ force identity for\n  shared tokens",
            edgecolor="#707070")
header_above(ax, postproc_center, POSTPROC_H, "(4) Post-Processing")
arrow(ax, (matrix_left + MATRIX_SIZE, corpus_center[1] + 0.3), (postproc_center[0] - POSTPROC_W / 2, postproc_center[1] - 0.3), color="black")

# ---- Final initialized embedding: header OUTSIDE, box shrunk ----
FINAL_W, FINAL_H = 2.4, 1.1
final_center = (postproc_center[0], corpus_center[1] - 1.15)
rounded_box(ax, final_center, FINAL_W, FINAL_H,
            "weighted average of\naligned source\nembeddings\n"
            "$\\mathrm{Emb}_t = D^{t,s} \\times \\mathrm{Emb}_s$",
            edgecolor="green")
header_above(ax, final_center, FINAL_H, "(5) Initialized Embedding")
arrow(ax, (postproc_center[0], postproc_center[1] - POSTPROC_H / 2), (final_center[0], final_center[1] + FINAL_H / 2 + 0.55), color="black")

# tighten bounds to actual content
ax.set_xlim(0.1, postproc_center[0] + FINAL_W / 2 + 0.3)
ax.set_ylim(0.3, tok_s_center[1] + 0.7)

plt.savefig("plots/energizar_pipeline.pdf", bbox_inches="tight", dpi=200)
print("Saved plots/energizar_pipeline.pdf")