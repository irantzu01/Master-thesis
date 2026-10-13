import matplotlib.pyplot as plt
import numpy as np

# Schematic (illustrative, not real GIZA++ output) probability trajectories
# for p(y | "etxe") across EM iterations, over the FULL corpus (not just
# the 3 sentences shown in Table X -- those are a small illustrative
# sample of a much larger pattern this plot represents).
iterations = np.array([0, 1, 2, 3, 4, 5])

# "casa" -- the dominant, most frequent translation: co-occurs consistently
# across many sentences in the corpus -> probability keeps rising.
p_casa = np.array([0.20, 0.34, 0.48, 0.60, 0.70, 0.78])

# "domicilio" -- a genuine synonym, closer in meaning than "edificio" but
# less common than "casa": stabilizes at a real, moderate plateau.
p_domicilio = np.array([0.15, 0.16, 0.15, 0.13, 0.12, 0.11])

# "edificio" -- also a genuine, valid translation, but rarer still:
# stabilizes at a small but non-zero plateau, NOT toward zero.
p_edificio = np.array([0.15, 0.10, 0.07, 0.05, 0.04, 0.035])

# Pure incidental co-occurrence (an unrelated word, no real translation
# relationship) -- this is the one that decays toward zero.
p_incidental = np.array([0.20, 0.12, 0.06, 0.03, 0.015, 0.008])

p_rest = 1 - (p_casa + p_domicilio + p_edificio + p_incidental)

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(iterations, p_casa, marker="o", linewidth=2.5, color="#2E8B57",
         label=r'$p(\mathit{casa} \mid \mathit{etxe})$ -- dominant translation')
ax.plot(iterations, p_domicilio, marker="o", linewidth=2.0, color="#4C72B0",
         linestyle="-.", label=r'$p(\mathit{domicilio} \mid \mathit{etxe})$ -- rare synonym')
ax.plot(iterations, p_edificio, marker="o", linewidth=2.0, color="#DD8452",
         linestyle="--", label=r'$p(\mathit{edificio} \mid \mathit{etxe})$ -- rare synonym')
ax.plot(iterations, p_incidental, marker="o", linewidth=1.8, color="#C0504D",
         linestyle=":", label=r'$p(\mathit{coche} \mid \mathit{etxe})$ -- incidental noise')
ax.plot(iterations, p_rest, marker="o", linewidth=1.3, color="gray",
         linestyle=":", alpha=0.6, label="remaining vocabulary (combined)")

ax.set_xlabel("EM Iteration (over the full corpus)", fontsize=12)
ax.set_ylabel(r"Estimated probability $p(y \mid \mathit{etxe})$", fontsize=12)
ax.set_title("Schematic Convergence of GIZA++ Alignment Probabilities", fontsize=13)
ax.set_xticks(iterations)
ax.set_ylim(0, 1.0)

ax.legend(fontsize=9, loc="upper left", framealpha=0.95)

ax.grid(axis="y", linestyle=":", alpha=0.5)

# Plain text labels, no arrows, placed in the empty diagonal wedge that
# opens up between the rising "casa" line and the declining lower cluster
# as iterations progress -- checked against actual line values to avoid
# any overlap.
ax.text(4, 0.80, "consistent co-occurrence across the corpus\n$\\rightarrow$ probability keeps rising",
         fontsize=9, color="#2E8B57", ha="center", va="center")

ax.text(4, 0.40, "genuine but rarer synonyms stabilize\nat a real, non-zero plateau",
         fontsize=9, color="#4C72B0", ha="center", va="center")

ax.text(4, 0.25, "pure incidental noise\nfades toward zero",
         fontsize=9, color="#C0504D", ha="center", va="center")

plt.tight_layout()
plt.savefig("plots/giza_em_convergence.pdf", bbox_inches="tight", dpi=200)
print("Saved plots/giza_em_convergence.pdf")