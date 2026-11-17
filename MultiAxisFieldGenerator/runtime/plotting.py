"""Tool-shape preview used by the portable viewer."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collision import sample_abb_tool


def tool_plot(config, output):
    tool = sample_abb_tool(config)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection='3d')
    for group, name in enumerate(tool.names):
        xyz = tool.offsets_mm[tool.component_ids == group].numpy()
        ax.scatter(*xyz.T, s=8, label=name.replace('_', ' '))
    ax.scatter(0, 0, 0, color='black', marker='*', s=80, label='nozzle root')
    ax.set(xlabel='Local X (mm)', ylabel='Local Y (mm)', zlabel='Tool axis (mm)',
           title=f"{config.get('name', 'Tool')} enclosure")
    ax.set_box_aspect(np.maximum(np.ptp(tool.offsets_mm.numpy(), axis=0), 1e-3))
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'abb_tool.png', dpi=180)
    plt.close(fig)
