import numpy as np
import matplotlib.pyplot as plt

# Generate a mock EEG signal (sine wave + random noise)
fs = 250  # Sampling frequency in Hz
t = np.linspace(0, 3, fs * 3)  # 3 seconds of data
signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))  # 10Hz sine wave with noise

# Define windowing parameters
window_size = fs  # 1 second window (250 samples)
step_size = int(fs * 0.5)  # 0.5 second step size (125 samples)

# Calculate window positions
windows = []
for start in range(0, len(signal) - window_size + 1, step_size):
    end = start + window_size
    windows.append((start, end))

# Plot the signal
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label='EEG Signal', linewidth=1)
plt.title("EEG Signal with Overlapping Windows")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Add sliding windows to the plot
for i, (start, end) in enumerate(windows):
    plt.axvspan(t[start], t[end - 1], border_color=f"C{i % 10}", alpha=0.3, label=f"Window {i+1}" if i < 10 else "")

# Add legend and grid
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), ncol=1)
plt.grid()
plt.tight_layout()
plt.show()
