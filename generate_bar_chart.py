import matplotlib.pyplot as plt
import numpy as np

# Data directly matching Table 1 in main.tex
labels = ['Standard View', 'Difficult\n(Glare/Angle)', 'Ideal\n(Direct View)']
tm_scores = [0.65, 0.30, 0.85]
sg_scores = [0.88, 0.78, 0.96]
sift_scores = [0.85, 0.15, 1.00]
orb_scores = [0.45, 0.05, 0.80]
ecc_scores = [0.70, 0.00, 1.00]

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

# Set up the figure with a good aspect ratio for IEEE double columns
fig, ax = plt.subplots(figsize=(8, 5))

# Plot bars with distinct, professional colors
rects1 = ax.bar(x - width*2, tm_scores, width, label='Template Match', color='#f39c12')
rects2 = ax.bar(x - width, orb_scores, width, label='ORB', color='#9b59b6')
rects3 = ax.bar(x, sift_scores, width, label='SIFT', color='#3498db')
rects4 = ax.bar(x + width, ecc_scores, width, label='ECC', color='#2ecc71')
rects5 = ax.bar(x + width*2, sg_scores, width, label='SuperGlue', color='#e74c3c') # Highlight SG in red

# Add text for labels, title and custom x-axis tick labels
ax.set_ylabel('Confidence Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
ax.set_title('Robustness of Localization Algorithms Across Scenarios', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
ax.set_ylim(0, 1.1)

# Add subtle grid lines for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True) # Put grid behind bars

fig.tight_layout()

# Save the plot at high resolution suitable for publication
plt.savefig('confidence_chart.png', dpi=300, bbox_inches='tight')
print("Successfully saved 'confidence_chart.png'. You can now include it in your LaTeX file.")
