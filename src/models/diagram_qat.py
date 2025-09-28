# diagram_qat.py
# ------------------------------
# This script draws a simple flowchart of the QAT process.
# It's just for visualization and explanations for myself.
# ---

import matplotlib.pyplot as plt

# Each string is a stage in the QAT pipeline
steps = [
    "FP32 Model\n(normal training)",
    "Attach qconfig\n(target int8 backend)",
    "prepare_qat()\n(add fake quant ops)",
    "Train\n(fake int8 forward,\nFP32 backprop)",
    "convert()\n(real int8 model)"
]

# Create plot
fig, ax = plt.subplots(figsize=(10, 5))


# Draw steps as rounded boxes with arrows
for i, step in enumerate(steps):
    ax.text(i * 2, 0, step, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.4", fc="lightblue", ec="navy", lw=2),
            fontsize=10)

    # draw arrow to next step
    if i < len(steps) - 1:
        ax.annotate("", xy=(i * 2 + 1.2, 0), xytext=(i * 2 + 0.8, 0),
                    arrowprops=dict(arrowstyle="->", lw=2, color="black"))


# remove axes, add title
ax.axis("off")
plt.title("Quantization Aware Training Flow", fontsize=14, weight="bold")
plt.show()
