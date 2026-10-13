import matplotlib.pyplot as plt
import numpy as np

conditions = [
    "GPT-2 Spanish", "GPT-2 English",
    "Baseline", "CSCL-es", "CSCL-en",
    "TokAlign", "TokAlign-CSCL",
    "EnerGIZAr", "EnerGIZAr-CSCL",
]
accuracy = [0.875, 0.901, 0.960, 0.971, 0.872, 0.868, 0.864, 0.879, 0.912]

categories = [
    "Base Models", "Base Models",
    "CSCL Only", "CSCL Only", "CSCL Only",
    "CVA Only", "CVA + CSCL",
    "CVA Only", "CVA + CSCL",
]
color_map = {
    "Base Models": "#4C72B0",
    "CSCL Only":  "#E49E75",
    "CVA Only":   "#D0528B",
    "CVA + CSCL": "#55A868",
}
hatch_map = {
    "Base Models": "",
    "CSCL Only":  "//",
    "CVA Only":   "xx",
    "CVA + CSCL": "..",
}
bar_colors = [color_map[c] for c in categories]
bar_hatches = [hatch_map[c] for c in categories]

x = np.arange(len(conditions))

# Larger figure to give labels room to breathe at bigger font sizes
fig, ax = plt.subplots(figsize=(12, 6.5))

bars = ax.bar(x, accuracy, color=bar_colors, edgecolor="black", linewidth=0.8)
for bar, hatch in zip(bars, bar_hatches):
    bar.set_hatch(hatch)

# chance line
ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
ax.text(0.1, 0.515, "chance = 0.5", ha="left", va="bottom", fontsize=13, style="italic")

ax.set_xticks(x)
ax.set_xticklabels(conditions, rotation=40, ha="right", fontsize=13)
ax.set_ylabel("MultiBLiMP Accuracy (Basque)", fontsize=15)
ax.set_ylim(0.5, 1.03)   # starts at chance level (0.5), not 0 -- see caption note
ax.set_title("MultiBLiMP Zero-Shot Basque Accuracy by Condition", fontsize=16)
ax.tick_params(axis="y", labelsize=13)
ax.grid(axis="y", linestyle=":", alpha=0.5)

# value labels on top of each bar -- larger and bolder
for xi, val in zip(x, accuracy):
    ax.text(xi, val + 0.008, f"{val:.3f}", ha="center", va="bottom",
             fontsize=12, fontweight="bold")

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=color_map[c], edgecolor="black", hatch=hatch_map[c])
    for c in ["Base Models", "CSCL Only", "CVA Only", "CVA + CSCL"]
]
ax.legend(legend_handles, ["Base Models", "CSCL Only", "CVA Only", "CVA + CSCL"],
          loc="lower right", frameon=True, fontsize=12)

plt.tight_layout()
plt.savefig("plots/multiblimp_results.pdf", bbox_inches="tight")
print("Saved plots/multiblimp_results.pdf")