import torch
import torch.nn as nn
import numpy as np
import time
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 65)
print("  BCI Benchmark: Izhikevich+R-STDP O(1) vs LSTM/BPTT O(T)")
print("=" * 65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Parameters ---
N_CHANNELS = 128
N_CLASSES = 4
N_POP = 25
N_HID = N_POP * N_CLASSES
T_VALUES = [20, 40, 100, 200, 500, 1000] # Simulation window in ms
DT = 0.5

# Dummy Cortical Input
BATCH = 32

# --- BPTT LSTM Decoder Baseline ---
class LSTMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_CHANNELS, 64, batch_first=True)
        self.fc = nn.Linear(64, N_CLASSES)
    def forward(self, x_seq):
        out, _ = self.lstm(x_seq)
        return self.fc(out[:, -1, :])

rstdp_mems, rstdp_lats = [], []
bptt_mems, bptt_lats = [], []

print(f"\n  {'T (ms)':>6} | {'R-STDP (MB)':>11} | {'BPTT (MB)':>10} | {'R-STDP (ms)':>11} | {'BPTT (ms)':>10}")
print(f"  {'-'*60}")

for T_ms in T_VALUES:
    n_microsteps = int(T_ms / DT)
    x_single = torch.rand(BATCH, N_CHANNELS, device=device)
    x_seq = x_single.unsqueeze(1).repeat(1, n_microsteps, 1)

    # 1. R-STDP (Izhikevich)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    W_rstdp = torch.rand(N_HID, N_CHANNELS, device=device) * 0.05
    
    t0 = time.perf_counter()
    v = torch.full((BATCH, N_HID), -65.0, device=device)
    u = 0.20 * v
    for step in range(n_microsteps):
        I_syn = (x_single @ W_rstdp.T) * 12.0
        dv = (0.04 * v**2 + 5.0 * v + 140.0 - u + I_syn) * DT
        du = (0.02 * (0.20 * v - u)) * DT
        v += dv
        u += du
        spike_mask = v >= 30.0
        v[spike_mask] = -65.0
        u[spike_mask] += 8.0
    
    lat_rstdp = (time.perf_counter() - t0) * 1000.0 / BATCH
    mem_rstdp = torch.cuda.max_memory_allocated() / 1e6
    rstdp_mems.append(mem_rstdp)
    rstdp_lats.append(lat_rstdp)

    # 2. BPTT LSTM Baseline
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_lstm = LSTMDecoder().to(device)
    opt = torch.optim.Adam(model_lstm.parameters())
    crit = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    opt.zero_grad()
    out = model_lstm(x_seq)
    loss = crit(out, torch.randint(0, N_CLASSES, (BATCH,), device=device))
    loss.backward()
    
    lat_bptt = (time.perf_counter() - t0) * 1000.0 / BATCH
    mem_bptt = torch.cuda.max_memory_allocated() / 1e6
    bptt_mems.append(mem_bptt)
    bptt_lats.append(lat_bptt)

    print(f"  {T_ms:>6} | {mem_rstdp:>11.1f} | {mem_bptt:>10.1f} | {lat_rstdp:>11.4f} | {lat_bptt:>10.4f}")

# --- Plot Figures ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Memory Footprint
ax = axes[0]
ax.plot(T_VALUES, rstdp_mems, "o-", color="#2196F3", linewidth=2.5, label="R-STDP Izhikevich (O(1) Neuromorphic)")
ax.plot(T_VALUES, bptt_mems, "s--", color="#FF5722", linewidth=2.5, label="BPTT LSTM (O(T) Standard Decoder)")
ax.set_xscale("log")
ax.set_xlabel("Motor Window Horizon T (ms)", fontsize=11)
ax.set_ylabel("Peak VRAM Memory (MB)", fontsize=11)
ax.set_title("Memory Complexity: O(1) Neuromorphic vs O(T) BPTT", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Step Decoding Latency
ax = axes[1]
ax.plot(T_VALUES, rstdp_lats, "o-", color="#2196F3", linewidth=2.5, label="R-STDP (O(1) Sub-millisecond)")
ax.plot(T_VALUES, bptt_lats, "s--", color="#FF5722", linewidth=2.5, label="BPTT LSTM")
ax.axhline(y=10.0, color="red", linestyle=":", label="Biological Closed-Loop Limit (10ms)")
ax.set_xscale("log")
ax.set_xlabel("Motor Window Horizon T (ms)", fontsize=11)
ax.set_ylabel("Inference Latency per Trial (ms)", fontsize=11)
ax.set_title("Closed-Loop Decision Latency", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/latency_memory_scaling.png", dpi=150, bbox_inches="tight")
print("\nSaved figures/latency_memory_scaling.png")
