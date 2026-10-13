import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig = plt.figure(figsize=(11.5, 6.5))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0.1, 12.6)   # tightened to match actual content width, avoids excess right-side whitespace
ax.set_ylim(-0.15, 6.1)   # extra headroom at top so the shadow box (offset +0.15) isn't clipped
ax.axis("off")

BUBBLE_W, BUBBLE_H = 2.4, 0.9

def shadow_box(ax, center, width, height, offset=(0.15, 0.15)):
    """Draws a faded, offset rounded box BEHIND the main bubble, so a
    small sliver peeks out at one corner -- suggesting another language
    'behind' the visible one."""
    xy = (center[0] - width / 2 + offset[0], center[1] - height / 2 + offset[1])
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1, edgecolor="gray", facecolor="#e8e8e8", alpha=0.6, zorder=1,
    )
    ax.add_patch(box)

def rounded_box(ax, center, width, height, text, edgecolor, facecolor="white", fontsize=9):
    shadow_box(ax, center, width, height)
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

def arrow(ax, start, end, color="black"):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                          linewidth=1.6, color=color, zorder=2)
    ax.add_patch(a)

def arrow_label(ax, start, end, text, color="black", offset=(0, 0), fontsize=12, fontweight="bold"):
    mx, my = (start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1]
    ax.text(mx, my, text, ha="center", va="center", fontsize=fontsize,
             fontweight=fontweight, color=color, zorder=5,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

# ---- Corpus bubble (center column) ----
corpus_center = (2.6, 3.25)
rounded_box(ax, corpus_center, BUBBLE_W, BUBBLE_H,
            "\"Ahozko mintzamena\neskolan, kalean...\"",
            edgecolor="black", fontsize=9)
ax.text(corpus_center[0] - BUBBLE_W / 2 - 0.2, corpus_center[1], "Corpus",
         ha="right", va="center", fontsize=12, fontweight="bold")

# ---- Tokenizer_s bubble (directly above corpus, same x, same size) ----
tok_s_center = (corpus_center[0], corpus_center[1] + 1.9)
rounded_box(ax, tok_s_center, BUBBLE_W, BUBBLE_H,
            "4287, 19, 5502, 3, 88, 214...",
            edgecolor="#3B9DD6", fontsize=9)
arrow(ax, (corpus_center[0], corpus_center[1] + BUBBLE_H / 2),
          (tok_s_center[0], tok_s_center[1] - BUBBLE_H / 2 - 0.1))
ax.text(corpus_center[0] - 0.25, (corpus_center[1] + tok_s_center[1]) / 2,
         r"(1) Tokenizer$_s$", ha="right", va="center", fontsize=12, fontweight="bold")

# ---- Tokenizer_t bubble (directly below corpus, same x, same size) ----
tok_t_center = (corpus_center[0], corpus_center[1] - 1.9)
rounded_box(ax, tok_t_center, BUBBLE_W, BUBBLE_H,
            "791, 1648, 311, 636, 3940...",
            edgecolor="#E8A33D", fontsize=9)
arrow(ax, (corpus_center[0], corpus_center[1] - BUBBLE_H / 2),
          (tok_t_center[0], tok_t_center[1] + BUBBLE_H / 2 + 0.1))
ax.text(corpus_center[0] - 0.25, (corpus_center[1] + tok_t_center[1]) / 2,
         r"(1) Tokenizer$_t$", ha="right", va="center", fontsize=12, fontweight="bold")

# ---- Arrows from tokenizer bubbles to heatmaps: BLACK, bigger gap, small buffer before heatmap ----
GAP = 1.6            # increased spacing between tokenizer bubbles and heatmaps
TIP_BUFFER = 0.15    # small visible gap between arrow tip and heatmap edge
HEAT_SIZE = 1.6

heat_s_left = tok_s_center[0] + BUBBLE_W / 2 + GAP
heat_t_left = tok_t_center[0] + BUBBLE_W / 2 + GAP

arrow_start_s = (tok_s_center[0] + BUBBLE_W / 2, tok_s_center[1])
arrow_end_s = (heat_s_left - TIP_BUFFER, tok_s_center[1])
arrow(ax, arrow_start_s, arrow_end_s, color="black")
arrow_label(ax, arrow_start_s, arrow_end_s, "(2) GloVe", offset=(-0.35, 0.22))

arrow_start_t = (tok_t_center[0] + BUBBLE_W / 2, tok_t_center[1])
arrow_end_t = (heat_t_left - TIP_BUFFER, tok_t_center[1])
arrow(ax, arrow_start_t, arrow_end_t, color="black")
arrow_label(ax, arrow_start_t, arrow_end_t, "(2) GloVe", offset=(-0.35, 0.22))

# ---- GloVe heatmaps, with |V1| / |V2| axis scales matching the alignment matrix ----
np.random.seed(3)
n_vocab_ticks = 5  # matches the alignment matrix's 5-tick style: 0,1,2,...,|V|
heat_s = np.random.rand(n_vocab_ticks, 3) * 0.6 + 0.3
heat_t = np.random.rand(n_vocab_ticks, 3) * 0.6 + 0.3

ax_heat_s = ax.inset_axes([heat_s_left, tok_s_center[1] - HEAT_SIZE / 2, HEAT_SIZE, HEAT_SIZE], transform=ax.transData)
ax_heat_s.imshow(heat_s, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax_heat_s.set_xticks(range(3)); ax_heat_s.set_xticklabels(["0", "1", "2"], fontsize=8)
ax_heat_s.set_yticks(range(n_vocab_ticks))
ax_heat_s.set_yticklabels(["0", "1", "2", "...", r"$|V_1|$"], fontsize=8)
ax_heat_s.set_xlabel("Emb. Dim.", fontsize=7)
ax_heat_s.set_ylabel("Vocab Index", fontsize=7)
for spine in ax_heat_s.spines.values():
    spine.set_edgecolor("#3B9DD6"); spine.set_linewidth(2)

ax_heat_t = ax.inset_axes([heat_t_left, tok_t_center[1] - HEAT_SIZE / 2, HEAT_SIZE, HEAT_SIZE], transform=ax.transData)
ax_heat_t.imshow(heat_t, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
ax_heat_t.set_xticks(range(3)); ax_heat_t.set_xticklabels(["0", "1", "2"], fontsize=8)
ax_heat_t.set_yticks(range(n_vocab_ticks))
ax_heat_t.set_yticklabels(["0", "1", "2", "...", r"$|V_2|$"], fontsize=8)
ax_heat_t.set_xlabel("Emb. Dim.", fontsize=7)
ax_heat_t.set_ylabel("Vocab Index", fontsize=7)
for spine in ax_heat_t.spines.values():
    spine.set_edgecolor("#E8A33D"); spine.set_linewidth(2)

# ---- Cosine similarity arrows converging into the alignment matrix ----
align_left = heat_s_left + HEAT_SIZE + 1.4
ALIGN_SIZE = 2.3

cosine_start_top = (heat_s_left + HEAT_SIZE, tok_s_center[1])
cosine_end_top = (align_left, corpus_center[1] + 0.5)
arrow(ax, cosine_start_top, cosine_end_top, color="black")
arrow_label(ax, cosine_start_top, cosine_end_top, "(3) Cosine Sim.\n(every pair)", offset=(0, 0.32), fontsize=10)

cosine_start_bottom = (heat_t_left + HEAT_SIZE, tok_t_center[1])
cosine_end_bottom = (align_left, corpus_center[1] - 0.5)
arrow(ax, cosine_start_bottom, cosine_end_bottom, color="black")
arrow_label(ax, cosine_start_bottom, cosine_end_bottom, "(3) Cosine Sim.\n(every pair)", offset=(-0.3, 0.45), fontsize=10)

# ---- Alignment matrix: x-axis = |V1| (matches heat_s), y-axis = |V2| (matches heat_t) ----
n = 5
np.random.seed(7)
align = np.random.rand(n, n) * 0.5
highlight_cells = [(0, 0), (1, 2), (2, 0), (3, 1), (4, 3)]
for (r, c) in highlight_cells:
    align[r, c] = 0.85

ax_align = ax.inset_axes([align_left, corpus_center[1] - ALIGN_SIZE / 2, ALIGN_SIZE, ALIGN_SIZE], transform=ax.transData)
ax_align.imshow(align, cmap="Greys", vmin=0, vmax=1, aspect="auto")
ax_align.set_xticks(range(n))
ax_align.set_xticklabels(["0", "1", "2", "...", r"$|V_1|$"], fontsize=8)
ax_align.set_yticks(range(n))
ax_align.set_yticklabels(["0", "1", "2", "...", r"$|V_2|$"], fontsize=8)
ax_align.set_title(r"Alignment Matrix $M_{s\rightarrow t}$", fontsize=9)
for (r, c) in highlight_cells:
    rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                edgecolor="black", linewidth=2.2)
    ax_align.add_patch(rect)

# ---- Fix 3: legend explaining what a highlighted cell means ----
legend_y = corpus_center[1] - ALIGN_SIZE / 2 - 0.55
legend_swatch = mpatches.Rectangle((align_left, legend_y - 0.09), 0.22, 0.22,
                                     fill=False, edgecolor="black", linewidth=2.2)
ax.add_patch(legend_swatch)
ax.text(align_left + 0.35, legend_y + 0.02, "= best match selected (argmax)",
         ha="left", va="center", fontsize=9)

# ---- Fix 4: concrete example callout tying the matrix back to etxe/casa ----
# highlight_cells[0] = (0, 0) -- treat this as the etxe/casa match for illustration
example_row, example_col = highlight_cells[0]
cell_center_axfrac_x = (example_col + 0.5) / n
cell_center_axfrac_y = 1 - (example_row + 0.5) / n  # imshow row 0 is at the top
cell_x = align_left + cell_center_axfrac_x * ALIGN_SIZE
cell_y = (corpus_center[1] - ALIGN_SIZE / 2) + cell_center_axfrac_y * ALIGN_SIZE

callout_x, callout_y = align_left + ALIGN_SIZE + 0.35, corpus_center[1] + ALIGN_SIZE / 2 - 0.15
ax.annotate("e.g., etxe \u2194 casa",
             xy=(cell_x, cell_y), xytext=(callout_x, callout_y),
             fontsize=10, fontweight="bold", color="#2E8B57",
             ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color="#2E8B57", lw=1.4))

plt.savefig("plots/tokalign_pipeline.pdf", bbox_inches="tight", dpi=200)
print("Saved plots/tokalign_pipeline.pdf")