import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

order_macro = [
    # --- OFF-THE-SHELF BENCHMARK MODELS (Bottom Block) ---
    {"Label": "GPT-2 English", "PPL": 47.6422, "Color": "#e57373"},      
    {"Label": "GPT-2 English", "PPL": 855.2476, "Color": '#4fc3f7'},     
    {"Label": "GPT-2 Spanish", "PPL": 78.99, "Color": "#81c784"},        
    {"Label": "GPT-2 Spanish", "PPL": 1535.94, "Color": '#0d47a1'},      
    
    # --- MODELS TRAINED IN THIS WORK (Top Block) ---
    {"Label": "CSCL-en", "PPL": 17.76, "Color": "#4fc3f7"},              
    {"Label": "Baseline", "PPL": 16.47, "Color": '#0d47a1'}, 
    {"Label": "CSCL-es", "PPL": 17.76, "Color": "#0d47a1"}               
]

y_pos = [0.3, 0.9, 2.1, 2.7, 4.5, 5.7, 6.9]

# Compact layout matches the width of the accuracy plot
fig, ax = plt.subplots(figsize=(9.0, 5.8))
bars = ax.barh(y_pos, [x['PPL'] for x in order_macro], color=[x['Color'] for x in order_macro], height=0.55, edgecolor='none')
ax.set_xscale('log')

for bar, x in zip(bars, order_macro):
    width = bar.get_width()
    is_proposed = x['Label'] == "Baseline"  # Highlight the proposed model
    ax.text(
        width * 1.06, 
        bar.get_y() + bar.get_height()/2, 
        f'{width:.2f}',
        va='center', ha='left', fontsize=11, fontweight='bold' if is_proposed else 'normal'
    )

tick_positions = [0.6, 2.4, 4.5, 5.7, 6.9]
tick_labels = ["GPT-2 English", "GPT-2 Spanish", "CSCL-en", "Baseline", "CSCL-es"]
ax.set_yticks(tick_positions)
ax.set_yticklabels(tick_labels, fontsize=11)

ax.text(6, 7.5, 'MODELS TRAINED IN THIS WORK', fontsize=12, fontweight='bold', color='#0d47a1')
ax.text(6, 3.2, 'OFF-THE-SHELF BENCHMARK MODELS', fontsize=12, fontweight='bold', color='#455a64')
ax.axhline(y=3.6, xmin=0, xmax=0.98, color='#b0bec5', linestyle='-', linewidth=1)

legend_elements = [
    Patch(facecolor='#0d47a1', label='Evaluated on Basque, Spanish-Based'),
    Patch(facecolor='#4fc3f7', label='Evaluated on Basque, English-Based'),
    Patch(facecolor='#81c784', label='Evaluated on Spanish'),
    Patch(facecolor='#e57373', label='Evaluated on English')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

ax.set_xlim(5, 5000)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

# Selected specific logarithm milestones to compress label frequency and distance
ax.set_xticks([5, 20, 100, 500, 2500])
ax.tick_params(axis='x', labelsize=11)

ax.set_xlabel('Perplexity (PPL)↓', fontsize=12, labelpad=8)
ax.set_title('FLORES-200 Benchmark Perplexity Results', fontsize=14, pad=30, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#b0bec5')
ax.spines['bottom'].set_color('#b0bec5')
ax.grid(axis='x', linestyle=':', alpha=0.6, which='both')

plt.tight_layout()
plt.savefig('perplexity_plot.png', dpi=300)