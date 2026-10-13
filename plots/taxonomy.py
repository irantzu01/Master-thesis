import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_latex_taxonomy():
    # Set up figure and axis (12x6.5 inches for a clean aspect ratio)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(-0.2, 11.2)
    ax.set_ylim(-0.3, 6.2)
    ax.axis('off')

    # Color Palette matching TikZ styles
    COLORS = {
        'base': {
            'fill': '#EBF2FA', 'edge': '#1D4ED8', 'title': '#1E3A8A', 'sub': '#3B82F6'
        },
        'process': {
            'fill': '#FFF7ED', 'edge': '#EA580C', 'title': '#7C2D12', 'sub': '#C2410C'
        },
        'cva': {
            'fill': '#FAF5FF', 'edge': '#7E22CE', 'title': '#581C87', 'sub': '#A855F7'
        },
        'final': {
            'fill': '#F0FDF4', 'edge': '#15803D', 'title': '#14532D', 'sub': '#16A34A'
        }
    }

    # Helper function to draw rounded boxes
    def draw_box(x, y, w, h, title, subtitle, box_type):
        c = COLORS[box_type]
        rect = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2, edgecolor=c['edge'], facecolor=c['fill'],
            zorder=3
        )
        ax.add_patch(rect)
        
        if subtitle:
            ax.text(x, y + 0.1, title, ha='center', va='center', fontsize=11, fontweight='bold', color=c['title'], zorder=4)
            ax.text(x, y - 0.12, subtitle, ha='center', va='center', fontsize=10, color=c['sub'], zorder=4)
        else:
            ax.text(x, y, title, ha='center', va='center', fontsize=11.5, fontweight='bold', color=c['title'], zorder=4)

    # Helper function to draw arrows
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color="#374151", lw=1.2, 
                mutation_scale=10, shrinkA=0, shrinkB=0
            ),
            zorder=2
        )

    # ==========================================
    # 0. COLOR LEGEND (Top)
    # ==========================================
    legend_y = 5.7
    legend_items = [
        ("Base Models", "base"),
        ("Vocabulary Adaptation (CVA)", "cva"),
        ("Pre-training Configuration", "process"),
        ("Final Models", "final")
    ]
    
    # Outer legend bounding box
    legend_bg = patches.FancyBboxPatch(
        (0.2, legend_y - 0.25), 10.6, 0.5,
        boxstyle="round,pad=0,rounding_size=0.06",
        linewidth=0.8, edgecolor='#D1D5DB', facecolor='#F9FAFB', zorder=1
    )
    ax.add_patch(legend_bg)

    # Horizontal Spacing Control:
    # Option A: Manual spacing list per item [gap_0->1, gap_1->2, gap_2->3, ...]
    spacing = [2.2, 3.1, 2.9, 1.8]  
    
    lx = 0.5
    for i, (label, btype) in enumerate(legend_items):
        c = COLORS[btype]
        box = patches.FancyBboxPatch(
            (lx, legend_y - 0.1), 0.22, 0.2,
            boxstyle="round,pad=0,rounding_size=0.04",
            linewidth=1.0, edgecolor=c['edge'], facecolor=c['fill'], zorder=2
        )
        ax.add_patch(box)
        ax.text(lx + 0.32, legend_y, label, va='center', fontsize=10.5, color='#1F2937', zorder=2)
        
        # Advance position using the defined spacing
        lx += spacing[i]

    # Column Centers (X-Coordinates for 5 Columns)
    cols = [1.0, 3.2, 5.4, 7.6, 9.8]
    box_w = 1.8

    # ==========================================
    # 1. TOP LEVEL: BASE MODELS
    # ==========================================
    draw_box(cols[0], 4.5, box_w, 0.7, "GPT-2 English", "(124M / OpenAI)", 'base')
    
    # Spanish base model centered between column 2 and column 5
    es_mid_x = (cols[1] + cols[4]) / 2  # 6.5
    draw_box(es_mid_x, 4.5, box_w, 0.7, "GPT-2 Spanish", "(117M / DeepEsp)", 'base')

    # ==========================================
    # 2. MID LEVEL: PIPELINE COLUMNS
    # ==========================================
    # Column 1
    draw_box(cols[0], 2.6, box_w, 0.65, "3-Phase CSCL", "(English Curriculum)", 'process')
    
    # Column 2
    draw_box(cols[1], 2.6, box_w, 0.65, "Monolingual Baseline", "Basque Pre-training", 'process')
    
    # Column 3
    draw_box(cols[2], 2.6, box_w, 0.65, "3-Phase CSCL", "(Spanish Curriculum)", 'process')

    # Column 4 (CVA -> CSCL)
    draw_box(cols[3], 3.25, box_w, 0.55, "TokAlign", "Stabilized", 'cva')
    draw_box(cols[3], 2.15, box_w, 0.55, "3-Phase CSCL", "Pipeline", 'process')

    # Column 5 (CVA -> CSCL)
    draw_box(cols[4], 3.25, box_w, 0.55, "EnerGIZAr", "Untrained Init", 'cva')
    draw_box(cols[4], 2.15, box_w, 0.55, "3-Phase CSCL", "Pipeline", 'process')

    # ==========================================
    # 3. BOTTOM LEVEL: FINAL EVALUATED MODELS
    # ==========================================
    draw_box(cols[0], 0.6, box_w, 0.6, "CSCL-en", "", 'final')
    draw_box(cols[1], 0.6, box_w, 0.6, "Baseline", "", 'final')
    draw_box(cols[2], 0.6, box_w, 0.6, "CSCL-es", "", 'final')
    draw_box(cols[3], 0.6, box_w, 0.6, "TokAlign-CSCL", "", 'final')
    draw_box(cols[4], 0.6, box_w, 0.6, "EnerGIZAr-CSCL", "", 'final')

    # ==========================================
    # 4. CONNECTIONS & ARROWS
    # ==========================================
    # English Connections
    draw_arrow(cols[0], 4.15, cols[0], 2.925)
    draw_arrow(cols[0], 2.275, cols[0], 0.9)

    # Spanish Connections (1-to-4 Split Bus)
    bus_y = 3.85
    ax.plot([es_mid_x, es_mid_x], [4.15, bus_y], color="#374151", lw=1.2, zorder=2)
    
    # Horizontal bus bar span
    ax.plot([cols[1], cols[4]], [bus_y, bus_y], color="#374151", lw=1.2, zorder=2)

    # Downward arrows from bus
    draw_arrow(cols[1], bus_y, cols[1], 2.925)
    draw_arrow(cols[2], bus_y, cols[2], 2.925)
    draw_arrow(cols[3], bus_y, cols[3], 3.525)
    draw_arrow(cols[4], bus_y, cols[4], 3.525)

    # Process to Final Connections
    draw_arrow(cols[1], 2.275, cols[1], 0.9)
    draw_arrow(cols[2], 2.275, cols[2], 0.9)

    # CVA to CSCL Pipeline Connections
    draw_arrow(cols[3], 2.975, cols[3], 2.425)
    draw_arrow(cols[3], 1.875, cols[3], 0.9)

    draw_arrow(cols[4], 2.975, cols[4], 2.425)
    draw_arrow(cols[4], 1.875, cols[4], 0.9)

    # Save vector PDF output
    plt.tight_layout()
    plt.savefig('master_taxonomy.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("PDF generated successfully: master_taxonomy.pdf")

if __name__ == '__main__':
    draw_latex_taxonomy()