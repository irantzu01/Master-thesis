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

# model order: GPT-2 Spanish, GPT-2 English, Baseline, CSCL-es, CSCL-en,
# TokAlign-Untrained, TokAlign-CSCL, EnerGIZAr-Untrained, EnerGIZAr-CSCL
data = {
    "Word-Level Cosine": {
        "matched":     [0.6567, 0.9903, 0.7302, 0.7505, 0.6365, 0.6891, 0.6869, 0.6380, 0.8543],
        "matched_std": [0.0025, 0.0008, 0.0063, 0.0036, 0.0033, 0.0059, 0.0052, 0.0113, 0.0026],
        "random":      [0.5566, 0.9906, 0.6720, 0.6908, 0.6386, 0.6245, 0.6242, 0.5191, 0.8073],
        "random_std":  [0.0075, 0.0004, 0.0044, 0.0042, 0.0034, 0.0032, 0.0033, 0.0084, 0.0036],
    },
    "Word-Level Jaccard": {
        "matched":     [0.5005, 0.8292, 0.5824, 0.6062, 0.4623, 0.5332, 0.5261, 0.4892, 0.7468],
        "matched_std": [0.0031, 0.0160, 0.0082, 0.0052, 0.0044, 0.0078, 0.0071, 0.0124, 0.0039],
        "random":      [0.3929, 0.8309, 0.5081, 0.5282, 0.4635, 0.4568, 0.4510, 0.3666, 0.6771],
        "random_std":  [0.0068, 0.0070, 0.0047, 0.0045, 0.0043, 0.0044, 0.0045, 0.0074, 0.0049],
    },
    "Sentence-Level Cosine": {
        "matched":     [0.1651, 0.9813, 0.5899, 0.6720, 0.5589, 0.6382, 0.7114, 0.7836, 0.8734],
        "matched_std": [0.0094, 0.0002, 0.0072, 0.0045, 0.0037, 0.0032, 0.0021, 0.0029, 0.0020],
        "random":      [0.1204, 0.9811, 0.4872, 0.5628, 0.5157, 0.5123, 0.5863, 0.7422, 0.8372],
        "random_std":  [0.0006, 0.0000, 0.0009, 0.0012, 0.0013, 0.0016, 0.0014, 0.0004, 0.0003],
    },
    "Sentence-Level Jaccard": {
        "matched":     [0.0893, 0.9582, 0.4252, 0.5093, 0.7438, 0.4719, 0.5533, 0.6441, 0.7651],
        "matched_std": [0.0055, 0.0003, 0.0068, 0.0048, 0.0042, 0.0036, 0.0028, 0.0038, 0.0034],
        "random":      [0.0619, 0.9568, 0.3250, 0.3927, 0.7047, 0.3455, 0.4149, 0.5906, 0.7107],
        "random_std":  [0.0003, 0.0001, 0.0007, 0.0011, 0.0014, 0.0013, 0.0013, 0.0004, 0.0005],
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
        error_kw = dict(elinewidth=1.1, capsize=3, capthick=1.1, ecolor="black")

        bars_matched = ax.bar(x - width/2, vals["matched"], width, color=bar_colors,
                                edgecolor="black", linewidth=0.9,
                                yerr=vals["matched_std"], error_kw=error_kw)
        for bar, hatch in zip(bars_matched, bar_hatches):
            bar.set_hatch(hatch)

        bars_random = ax.bar(x + width/2, vals["random"], width, color=bar_colors,
                               edgecolor="black", linewidth=0.9, alpha=0.4,
                               yerr=vals["random_std"], error_kw=error_kw)
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