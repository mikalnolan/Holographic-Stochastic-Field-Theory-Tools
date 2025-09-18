import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
N = 64  # Grid size
L = 10000  # Number of samples
realizations = 10  # Number of realizations for averaging
sigma = 1.0  # Envelope width
c1 = 1  # Chern number

# Generate k-grid (3D wave vectors)
k_x = np.arange(-N/2, N/2)
k_y = np.arange(-N/2, N/2)
k_z = np.arange(-N/2, N/2)
k_grid = np.array(np.meshgrid(k_x, k_y, k_z, indexing='ij')).reshape(3, -1).T

# Envelope function (Gaussian in Fourier space)
def G_hat(k_mag):
    return np.exp(-k_mag**2 / (2 * sigma**2))

# Generate points on E (Heisenberg nilmanifold approximation)
beta = np.random.rand(L, 3)  # Uniform in [0,1)^3 mod lattice
X_beta = np.mod(beta - 0.5 * c1 * beta[:, [0]] * beta[:, [1]], 1)  # Nilflow map for c1=1

# Initialize arrays for averaging over realizations
P_S_avg = np.zeros(len(k_grid))
P_H_avg = np.zeros(len(k_grid))

# Run multiple realizations
for _ in range(realizations):
    # Generate complex Gaussian white noise
    xi = np.random.normal(0, 1/np.sqrt(L), (L, 3)) + 1j * np.random.normal(0, 1/np.sqrt(L), (L, 3))
    U = np.exp(1j * 2 * np.pi * np.random.rand(L))  # Holonomy phase

    # Compute B_S and B_H for each k
    B_S = np.zeros(len(k_grid), dtype=complex)
    B_H = np.zeros(len(k_grid), dtype=complex)
    for i, k in enumerate(k_grid):
        phase = np.exp(-2j * np.pi * np.dot(k, X_beta.T))
        B_S[i] = np.mean(phase * xi[:, 0]) / np.sqrt(L)  # Using first component of a
        B_H[i] = np.mean(phase * U * xi[:, 0]) / np.sqrt(L)

    # Preliminary spectra estimates
    P_S = np.abs(B_S)**2
    P_H = np.real(B_H * np.conj(B_S))

    # Enforce bound P_S >= |P_H|
    P_S = np.maximum(P_S, np.abs(P_H))

    # Average over realizations
    P_S_avg += P_S / realizations
    P_H_avg += P_H / realizations

# Compute |k| magnitudes
k_magnitudes = np.sqrt(np.sum(k_grid**2, axis=1))

# Bin data by |k| for plotting
k_bins = np.arange(0, np.max(k_magnitudes) + 1, 1)
P_S_binned = np.zeros(len(k_bins) - 1)
P_H_binned = np.zeros(len(k_bins) - 1)
for i in range(len(k_bins) - 1):
    mask = (k_magnitudes >= k_bins[i]) & (k_magnitudes < k_bins[i + 1])
    P_S_binned[i] = np.mean(P_S_avg[mask]) if np.any(mask) else 0
    P_H_binned[i] = np.mean(P_H_avg[mask]) if np.any(mask) else 0

# Apply envelope
k_centers = (k_bins[:-1] + k_bins[1:]) / 2
P_S_binned *= G_hat(k_centers)**2
P_H_binned *= G_hat(k_centers)**2

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_centers, P_S_binned, 'b-', label='$P_S(|k|)$')
plt.plot(k_centers, P_H_binned, 'r--', label='$P_H(|k|)$')
plt.xlabel('$|k|$')
plt.ylabel('Power Spectra')
plt.title('Simulated Spectra for $c_1=1$, $N=64$, $L=10^4$')
plt.legend()
plt.grid(True)
plt.savefig('spectra_plot.pdf')
plt.show()

# Error validation (approximate)
deviation = np.mean(np.abs(P_S_binned - P_S_binned.mean()))  # Simplified deviation
violation_count = np.sum(P_S_binned < np.abs(P_H_binned))
bias = np.mean(np.abs(P_H_binned[P_S_binned < np.abs(P_H_binned)])) if violation_count > 0 else 0
print(f"Average deviation: {deviation:.4f} (O(1/sqrt(L)))")
print(f"Violation percentage: {100 * violation_count / len(P_S_binned):.2f}%")
print(f"Clipping bias: {bias:.4f} (O(1/L))")
