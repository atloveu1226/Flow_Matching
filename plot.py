import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.titlesize': 20
})


data_dir = "data/eulerian"
common_steps = []
w2_vals = []
npe_vals = []
for file in os.listdir(data_dir):
    if file.endswith(".npz"):
        file_path = os.path.join(data_dir, file)
        key = file.split(".")[0].split("_")[-1]
        print(f"Processing key: {key}")
        data = np.load(file_path)
        steps = data['steps']
        w2 = data['w2']
        npe = data['npe']
        
        common_steps = steps
        w2_vals.append(w2)
        npe_vals.append(npe)
        
        
w2_vals = np.array(w2_vals)
npe_vals = np.array(npe_vals)
mean_w2 = np.mean(w2_vals, axis=0)
std_w2 = np.std(w2_vals, axis=0)
mean_npe = np.mean(npe_vals, axis=0)
std_npe = np.std(npe_vals, axis=0)
    

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# w2 plot
axes[0].plot(common_steps, mean_w2, label='Mean W2', color='navy', linewidth=2)
axes[0].fill_between(common_steps, mean_w2 - std_w2, mean_w2 + std_w2, color='navy', alpha=0.2)
axes[0].set_xlabel('Steps')
axes[0].set_ylabel('W2')
axes[0].set_title('W2 vs Steps')
axes[0].legend(frameon=True)

# npe plot
axes[1].plot(common_steps, mean_npe, label='Mean NPE', color='darkred', linewidth=2)
axes[1].fill_between(common_steps, mean_npe - std_npe, mean_npe + std_npe, color='darkred', alpha=0.2)
axes[1].set_xlabel('Steps')
axes[1].set_ylabel('NPE')
axes[1].set_title('NPE vs Steps')
axes[1].legend(frameon=True)

fig.suptitle('Flow Matching Metrics', fontsize=20)
fig.tight_layout(rect=[0, 0, 1, 0.96])


output_path = os.path.join(os.path.dirname(__file__), "../plots", "flow_matching_metrics.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)