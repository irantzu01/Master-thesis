import matplotlib.pyplot as plt
import numpy as np

models = ["Baseline", "CSCL-es", "CSCL-en", "TokAlign", "TokAlign-CSCL",
          "EnerGIZAr-Untrained", "EnerGIZAr-CSCL"]

categories = ["CSCL Only", "CSCL Only", "CSCL Only",
              "CVA Only", "CVA + CSCL",
              "CVA Only", "CVA + CSCL"]

color_map = {
    "CSCL Only":  "#DD8452",
    "CVA Only":   "#8172B2",
    "CVA + CSCL": "#55A868",
}
hatch_map = {
    "CSCL Only":  "//",
    "CVA Only":   "xx",
    "CVA + CSCL": "..",
}

tasks = {
    "BEC":   {"values": [0.337, 0.334, 0.334, 0.334, 0.313, 0.346, 0.339], "chance": 0.333, "ymax": 0.42},
    "BHTC":  {"values": [0.100, 0.125, 0.101, 0.117, 0.185, 0.078, 0.010], "chance": 0.083, "ymax": 0.22},
    "Coref": {"values": [0.477, 0.509, 0.472, 0.503, 0.496, 0.503, 0.497], "chance": 0.500, "ymax": 0.60},
    "QNLI":  {"values": [0.433, 0.504, 0.441, 0.500, 0.458, 0.500, 0.500], "chance": 0.500, "ymax": 0.60},
    "Vaxx":  {"values": [0.261, 0.289, 0.303, 0.279, 0.066, 0.226, 0.275], "chance": 0.500, "ymax": 0.60},
    "WiC":   {"values": [0.499, 0.496, 0.494, 0.499, 0.497, 0.501, 0.500], "chance": 0.500, "ymax": 0.60},
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
axes = axes.flatten()
x = np.arange(len(models))
bar_colors = [color_map[c] for c in categories]
bar_hatches = [hatch_map[c] for c in categories]

for ax, (task_name, info) in zip(axes, tasks.items()):
    bars = ax.bar(x, info["values"], color=bar_colors, edgecolor="black", linewidth=0.8)
    for bar, hatch in zip(bars, bar_hatches):
        bar.set_hatch(hatch)

    ax.axhline(info["chance"], color="black", linestyle="--", linewidth=1.2, alpha=0.8,
                label="chance" if task_name == "BEC" else None)

    for xi, val in zip(x, info["values"]):
        ax.text(xi, val + info["ymax"] * 0.015, f"{val:.3f}", ha="center", va="bottom",
                 fontsize=7.5, fontweight="bold")

    ax.set_title(task_name, fontsize=13, fontweight="bold")
    ax.set_ylim(0, info["ymax"])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=color_map[c], edgecolor="black", hatch=hatch_map[c])
    for c in ["CSCL Only", "CVA Only", "CVA + CSCL"]
]
legend_handles.append(plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.2))
legend_labels = ["CSCL Only", "CVA Only", "CVA + CSCL", "Chance level (task-specific)"]
fig.legend(legend_handles, legend_labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4, fontsize=10.5, frameon=False)

fig.suptitle("BasqueGLUE 5-Shot Results by Task", fontsize=15, y=1.06)
plt.tight_layout()
plt.savefig("plots/basqueglue_results.pdf", bbox_inches="tight", dpi=200)
print("Saved plots/basqueglue_results.pdf")