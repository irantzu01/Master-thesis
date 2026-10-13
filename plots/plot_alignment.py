import matplotlib.pyplot as plt
import numpy as np

models = ["GPT-2 Spanish", "GPT-2 English", "Baseline", "CSCL-es", "CSCL-en",
          "TokAlign-Untrained", "TokAlign-CSCL", "EnerGIZAr-Untrained", "EnerGIZAr-CSCL"]

categories = ["Base Model", "Base Model", "CSCL Only", "CSCL Only", "CSCL Only",
              "CVA Only", "CVA + CSCL", "CVA Only", "CVA + CSCL"]

color_map = {
    "Base Model": "#4C72B0",
    "CSCL Only":  "#DD8452",
    "CVA Only":   "#8172B2",
    "CVA + CSCL": "#55A868",
}
hatch_map = {
    "Base Model": "",
    "CSCL Only":  "//",
    "CVA Only":   "xx",
    "CVA + CSCL": "..",
}

data = {
    "Word-Level Cosine": {
        "matched": [0.6567, 0.9903, 0.7302, 0.7505, 0.6365, 0.6891, 0.6869, 0.6380, 0.8543],
        "random":  [0.5566, 0.9906, 0.6720, 0.6908, 0.6386, 0.6245, 0.6242, 0.5191, 0.8073],
    },
    "Word-Level Jaccard": {
        "matched": [0.5005, 0.8292, 0.5824, 0.6062, 0.4623, 0.5332, 0.5261, 0.4892, 0.7468],
        "random":  [0.3929, 0.8309, 0.5081, 0.5282, 0.4635, 0.4568, 0.4510, 0.3666, 0.6771],
    },
    "Sentence-Level Cosine": {
        "matched": [0.1651, 0.9813, 0.5899, 0.6720, 0.5589, 0.6382, 0.7114, 0.7836, 0.8734],
        "random":  [0.1204, 0.9811, 0.4872, 0.5628, 0.5157, 0.5123, 0.5863, 0.7422, 0.8372],
    },
    "Sentence-Level Jaccard": {
        "matched": [0.0893, 0.9582, 0.4252, 0.5093, 0.7438, 0.4719, 0.5533, 0.6441, 0.7651],
        "random":  [0.0619, 0.9568, 0.3250, 0.3927, 0.7047, 0.3455, 0.4149, 0.5906, 0.7107],
    },
}


def plot_stacked_pair(panel_names, output_path, suptitle):
    fig, axes = plt.subplots(2, 1, figsize=(11, 11))
    x = np.arange(len(models))
    width = 0.38
    bar_colors = [color_map[c] for c in categories]
    bar_hatches = [hatch_map[c] for c in categories]

    for ax, panel_name in zip(axes, panel_names):
        vals = data[panel_name]
        bars_matched = ax.bar(x - width/2, vals["matched"], width, color=bar_colors,
                                edgecolor="black", linewidth=0.9)
        for bar, hatch in zip(bars_matched, bar_hatches):
            bar.set_hatch(hatch)

        bars_random = ax.bar(x + width/2, vals["random"], width, color=bar_colors,
                               edgecolor="black", linewidth=0.9, alpha=0.4)
        for bar, hatch in zip(bars_random, bar_hatches):
            bar.set_hatch(hatch)

        ax.set_title(panel_name, fontsize=15, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=40, ha="right", fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    category_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map[c], edgecolor="black", hatch=hatch_map[c])
        for c in ["Base Model", "CSCL Only", "CVA Only", "CVA + CSCL"]
    ]
    opacity_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="black", alpha=1.0),
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="black", alpha=0.4),
    ]
    fig.legend(category_handles + opacity_handles,
                ["Base Model", "CSCL Only", "CVA Only", "CVA + CSCL", "Matched pair", "Random baseline"],
                loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize=12, frameon=False)

    fig.suptitle(suptitle, fontsize=16, y=1.10)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {output_path}")


plot_stacked_pair(
    ["Word-Level Cosine", "Word-Level Jaccard"],
    "plots/alignment_word_level.pdf",
    "Word-Level Alignment: Matched vs. Random-Pair Baseline",
)

plot_stacked_pair(
    ["Sentence-Level Cosine", "Sentence-Level Jaccard"],
    "plots/alignment_sentence_level.pdf",
    "Sentence-Level Alignment: Matched vs. Random-Pair Baseline",
)