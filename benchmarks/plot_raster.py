import torch
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 65)
print("  Generating Izhikevich Spiking Dynamics & Raster Plot")
print("=" * 65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CHANNELS = 128
N_CLASSES = 4
N_POP = 25
N_HID = 100
T_STEPS = 60.0
DT = 0.5
N_STEPS = int(T_STEPS / DT)

torch.manual_seed(42)
W = torch.rand(N_HID, N_CHANNELS, device=device) * 0.1
x_sample = torch.rand(N_CHANNELS, device=device)

v = torch.full((N_HID,), -65.0, device=device)
u = 0.20 * v

v_history = []
spikes_history = []

for step in range(N_STEPS):
    I_syn = (W @ x_sample) * 15.0
    dv = (0.04 * v**2 + 5.0 * v + 140.0 - u + I_syn) * DT
    du = (0.02 * (0.20 * v - u)) * DT
    v += dv
    u += du
    
    spike_mask = v >= 30.0
    spikes_history.append(spike_mask.cpu().numpy())
    v_history.append(v[0].item()) # Track neuron 0 membrane voltage
    
    v[spike_mask] = -65.0
    u[spike_mask] += 8.0

spikes_matrix = np.array(spikes_history).T # (N_HID, N_STEPS)
time_axis = np.arange(N_STEPS) * DT

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Plot 1: Membrane Voltage of Single Izhikevich RS Neuron
ax1 = axes[0]
ax1.plot(time_axis, v_history, color="#00E676", linewidth=1.8, label="Izhikevich RS Neuron (v)")
ax1.axhline(y=30.0, color="red", linestyle="--", alpha=0.6, label="Threshold (30 mV)")
ax1.set_ylabel("Membrane Voltage (mV)", fontsize=11)
ax1.set_title("Izhikevich (2003) Regular Spiking Membrane Potential Dynamics", fontsize=12, fontweight="bold")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# Plot 2: Cortical Population Raster Plot
ax2 = axes[1]
colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
directions = ["Up (0-24)", "Down (25-49)", "Left (50-74)", "Right (75-99)"]

for c in range(N_CLASSES):
    pop_spikes = spikes_matrix[c*N_POP:(c+1)*N_POP, :]
    neuron_idx, step_idx = np.where(pop_spikes)
    ax2.scatter(step_idx * DT, neuron_idx + c*N_POP, s=12, color=colors[c], label=f"Pop {directions[c]}")

ax2.set_xlabel("Time (ms)", fontsize=11)
ax2.set_ylabel("Izhikevich Neuron Index", fontsize=11)
ax2.set_title("100-Neuron Cortical Intent Population Raster Plot", fontsize=12, fontweight="bold")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/izhikevich_raster.png", dpi=150, bbox_inches="tight")
print("Saved figures/izhikevich_raster.png")
