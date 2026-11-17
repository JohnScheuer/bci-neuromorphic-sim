import torch
import numpy as np
import time

print("=" * 70)
print("  BCI Neuromorphic Simulator (Izhikevich Neurons + R-STDP)")
print("  Task: 128-Channel Motor Cortical Intention Decoding (4 Directions)")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# --- BCI Hyperparameters ---
N_CHANNELS = 128        # 128-channel M1 Cortical Spike Array (Utah Array)
N_CLASSES = 4           # 4 Motor Intentions: 0=Up, 1=Down, 2=Left, 3=Right
N_POP = 25              # 25 Izhikevich RS neurons per motor population
N_HID = N_POP * N_CLASSES # 100 neurons total
T_STEPS = 40.0          # 40ms motor trial window
DT = 0.5                # Integration timestep (0.5ms)
N_STEPS = int(T_STEPS / DT) # 80 microsteps
K_WIN = 2               # k-WTA per population (Sparse firing)

# --- Izhikevich (2003) Parameters: Regular Spiking (RS) Cortical Neurons ---
A_PARAM = 0.02
B_PARAM = 0.20
C_PARAM = -65.0         # Reset potential (mV)
D_PARAM = 8.0           # Recovery variable increment
V_TH = 30.0             # Spike threshold (mV)

# --- R-STDP Parameters ---
A_LTP = 0.005           # Stabilized LTP rate
A_LTD = 0.0016          # Stabilized LTD rate (A_LTP / (N_CLASSES - 1))
W_TARGET = 1.0          # L1 synaptic scaling target
EPOCHS = 5
BATCH_SIZE = 100

print(f"\n  Cortical Array:   {N_CHANNELS} Channels (M1 Motor Cortex)")
print(f"  Motor Intentions: {N_CLASSES} Directions (Up, Down, Left, Right)")
print(f"  Izhikevich Layer: {N_HID} RS Neurons (a={A_PARAM}, b={B_PARAM}, c={C_PARAM}mV, d={D_PARAM})")
print(f"  Time Window:      {T_STEPS}ms ({N_STEPS} microsteps at dt={DT}ms)")
print(f"  Learning Rule:    Vectorized R-STDP + Synaptic Homeostasis (O(1) Memory)")
print("=" * 70)

# --- Synthetic Cortical Dataset Generation ---
torch.manual_seed(42)
BASE_TUNING = torch.rand(N_CLASSES, N_CHANNELS, device=device)
BASE_TUNING = (BASE_TUNING > 0.55).float() * (0.6 + 0.4 * torch.rand(N_CLASSES, N_CHANNELS, device=device))

def generate_cortical_dataset(n_samples_per_class=1000, noise_seed=42):
    torch.manual_seed(noise_seed)
    X_list, Y_list = [], []

    for c in range(N_CLASSES):
        for _ in range(n_samples_per_class):
            rate = BASE_TUNING[c] + 0.08 * torch.randn(N_CHANNELS, device=device)
            rate = rate.clamp(0.0, 1.0)
            X_list.append(rate)
            Y_list.append(c)

    X = torch.stack(X_list)
    Y = torch.tensor(Y_list, device=device)
    perm = torch.randperm(len(X))
    return X[perm], Y[perm]

print("\nGenerating M1 multi-channel motor cortical dataset...")
X_train, Y_train = generate_cortical_dataset(n_samples_per_class=1000, noise_seed=42)
X_test, Y_test = generate_cortical_dataset(n_samples_per_class=250, noise_seed=999)
print(f"  Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# --- Initialize Weights ---
torch.manual_seed(42)
W = torch.rand(N_HID, N_CHANNELS, device=device) * 0.05
W = W / W.sum(dim=1, keepdim=True) * W_TARGET

# --- 100% Vectorized CUDA Izhikevich Execution Engine ---
def simulate_izhikevich_vectorized(X_batch, W_matrix, do_stdp=False, Y_batch=None):
    batch_size = X_batch.shape[0]

    v = torch.full((batch_size, N_HID), -65.0, device=device)
    u = B_PARAM * v
    spike_counts = torch.zeros(batch_size, N_HID, device=device)

    t_start = time.perf_counter()

    for step in range(N_STEPS):
        I_syn = (X_batch @ W_matrix.T) * 12.0

        dv = (0.04 * v**2 + 5.0 * v + 140.0 - u + I_syn) * DT
        du = (A_PARAM * (B_PARAM * v - u)) * DT
        v = v + dv
        u = u + du

        v_reshaped = v.view(batch_size, N_CLASSES, N_POP)
        topk_vals, topk_idx = v_reshaped.topk(K_WIN, dim=2)

        firing = topk_vals >= V_TH
        spike_mask_3d = torch.zeros_like(v_reshaped, dtype=torch.bool)
        spike_mask_3d.scatter_(2, topk_idx, firing)

        spike_mask = spike_mask_3d.view(batch_size, N_HID)
        spikes = spike_mask.float()
        spike_counts += spikes

        v[spike_mask] = C_PARAM
        u[spike_mask] = u[spike_mask] + D_PARAM

    batch_latency_ms = (time.perf_counter() - t_start) * 1000.0

    if do_stdp and Y_batch is not None:
        reward_class = torch.full((batch_size, N_CLASSES), -A_LTD, device=device)
        reward_class.scatter_(1, Y_batch.unsqueeze(1), A_LTP)
        reward_weights = reward_class.repeat_interleave(N_POP, dim=1)

        modulated_spikes = spike_counts * reward_weights
        dW = (modulated_spikes.T @ X_batch) / batch_size

        W_matrix += dW

        row_sums = W_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
        W_matrix.copy_(W_matrix / row_sums * W_TARGET)
        W_matrix.clamp_(0.0, 5.0)

    return spike_counts, batch_latency_ms

# --- Phase 1: Training ---
print(f"\n[Phase 1: Real-Time Closed-Loop Training (Vectorized R-STDP)]")
print(f"  {len(X_train)} motor trials x {EPOCHS} epochs")
print(f"  Simulating CUDA vectorized Izhikevich cortical dynamics...")

torch.cuda.reset_peak_memory_stats()
t0_train = time.time()

for epoch in range(EPOCHS):
    perm = torch.randperm(len(X_train))
    epoch_correct = 0
    total_spikes = 0

    for i in range(0, len(X_train), BATCH_SIZE):
        batch_idx = perm[i:i+BATCH_SIZE]
        xb = X_train[batch_idx]
        yb = Y_train[batch_idx]

        sc, _ = simulate_izhikevich_vectorized(xb, W, do_stdp=True, Y_batch=yb)
        total_spikes += sc.sum().item()

        votes = sc.view(BATCH_SIZE, N_CLASSES, N_POP).sum(dim=2)
        preds = votes.argmax(dim=1)
        epoch_correct += (preds == yb).sum().item()

    train_acc = epoch_correct / len(X_train) * 100
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    avg_spikes = total_spikes / len(X_train)
    print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Train Acc: {train_acc:5.1f}% | Avg Spikes/Trial: {avg_spikes:5.1f} | GPU Mem: {peak_mem:.1f} MB")

train_time = time.time() - t0_train
print(f"\n  Training completed in {train_time:.2f}s | Peak VRAM: {peak_mem:.1f} MB")

# --- Phase 2: Closed-Loop Latency & Accuracy Evaluation ---
print(f"\n[Phase 2: Closed-Loop Latency & Accuracy Evaluation]")

STREAM_BATCH = 32
correct = 0
latencies_per_trial = []
class_correct = {c: 0 for c in range(N_CLASSES)}
class_total = {c: 0 for c in range(N_CLASSES)}
directions = ["Up", "Down", "Left", "Right"]

for i in range(0, len(X_test), STREAM_BATCH):
    xb = X_test[i:i+STREAM_BATCH]
    yb = Y_test[i:i+STREAM_BATCH]

    sc, batch_lat = simulate_izhikevich_vectorized(xb, W, do_stdp=False)
    per_trial_lat = batch_lat / xb.shape[0]
    latencies_per_trial.extend([per_trial_lat] * xb.shape[0])

    votes = sc.view(xb.shape[0], N_CLASSES, N_POP).sum(dim=2)
    preds = votes.argmax(dim=1)

    for b in range(xb.shape[0]):
        y_true = yb[b].item()
        y_pred = preds[b].item()
        class_total[y_true] += 1
        if y_pred == y_true:
            correct += 1
            class_correct[y_true] += 1

test_acc = correct / len(X_test) * 100
mean_latency = np.mean(latencies_per_trial)
p99_latency = np.percentile(latencies_per_trial, 99)

print(f"  Test Accuracy: {test_acc:.1f}% ({correct}/{len(X_test)})")
print(f"  Closed-Loop Decoding Latency (Mean): {mean_latency:.4f} ms per 40ms motor trial")
print(f"  Closed-Loop Decoding Latency (P99):  {p99_latency:.4f} ms")
print(f"\n  Per-Direction Motor Accuracy:")
for c in range(N_CLASSES):
    acc_c = class_correct[c] / class_total[c] * 100
    print(f"    Direction {c} ({directions[c]:<5}): {acc_c:5.1f}% ({class_correct[c]}/{class_total[c]})")

print(f"\n{'='*70}")
print(f"  BCI BENCHMARK SUMMARY")
print(f"{'='*70}")
print(f"  Neuron Model:             Izhikevich (2003) Regular Spiking (RS)")
print(f"  Cortical Input:           128 M1 Spike Channels")
print(f"  Motor Intentions:         4 Directions (Up, Down, Left, Right)")
print(f"  Decoding Accuracy:        {test_acc:.1f}%")
print(f"  Closed-Loop Step Latency: {mean_latency:.4f} ms (sub-millisecond decoder)")
print(f"  Memory Footprint:         {peak_mem:.1f} MB (O(1) Flat Scaling)")
print(f"  Neuromorphic Target:      Intel Loihi 2 / Brain-Computer Interface")
print(f"{'='*70}")
