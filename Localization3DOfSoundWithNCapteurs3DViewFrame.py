# Author(s): Dr. Patrick Lemoine

import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import random

# Spherical to Cartesian coordinates
def sph_to_cart(r, theta_deg, phi_deg):
    theta = np.radians(theta_deg)  # inclination from +Z axis
    phi = np.radians(phi_deg)      # azimuth from +X axis
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

# Generate mic positions on sphere with Fibonacci method
def generate_mic_positions_on_sphere(n_mics, radius):
    indices = np.arange(0, n_mics, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_mics)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])

# Bandpass filter
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

# Simulate impulse signals for source located at given direction and distance factor
def simulate_impulse_signals(mic_positions, fs, duration_s,
                             src_theta, src_phi, c=343.0,
                             noise=0.01, source_distance_factor=1.5):
    n_mics = mic_positions.shape[0]
    radius = np.linalg.norm(mic_positions[0])
    src_distance = source_distance_factor * radius
    src_pos = sph_to_cart(src_distance, src_theta, src_phi)
    delays = []
    for mic_pos in mic_positions:
        distance = np.linalg.norm(src_pos - mic_pos)
        delays.append(distance / c)
    min_delay = min(delays)
    n_samples = int(duration_s * fs)
    signals = []
    for delay in delays:
        delay_samples = int((delay - min_delay) * fs)
        sig = np.zeros(n_samples)
        if delay_samples < n_samples:
            sig[delay_samples] = 1.0
        sig += noise * np.random.randn(n_samples)
        signals.append(sig)
    return signals, src_pos

# GCC-PHAT between two signals
def gcc_phat(sig1, sig2, fs, max_tau=None, interp=16):
    n = sig1.shape[0] + sig2.shape[0]
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    R /= (np.abs(R) + 1e-10)  # PHAT
    cc = np.fft.irfft(R, n=interp * n)
    max_shift = int(interp * fs * max_tau) if max_tau else int(len(cc) / 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    tau = np.linspace(-max_tau, max_tau, len(cc)) if max_tau else np.linspace(-0.5, 0.5, len(cc))
    max_idx = np.argmax(np.abs(cc))
    return tau[max_idx], cc[max_idx]

# Full SRP-PHAT beamforming on given frame signals
def srp_phat_full(signals, mic_positions, fs, grid_theta, grid_phi, c=343.0):
    n_mics = len(signals)
    radius = np.linalg.norm(mic_positions[0])
    max_tau = 2 * radius / c
    pairs = [(i, j) for i in range(n_mics) for j in range(i + 1, n_mics)]
    results = []
    for th, ph in itertools.product(grid_theta, grid_phi):
        steer_vec = sph_to_cart(radius * 3, th, ph)
        power = 0
        for i, j in pairs:
            mic_i = mic_positions[i]
            mic_j = mic_positions[j]
            expected_tau = (np.linalg.norm(steer_vec - mic_i) - np.linalg.norm(steer_vec - mic_j)) / c
            tau, val = gcc_phat(signals[i], signals[j], fs, max_tau=max_tau)
            weight = np.exp(-((tau - expected_tau) ** 2) / 0.001)
            power += np.abs(val) * weight
        results.append((power, th, ph))
    return results

# Select top k directions avoiding duplicates closer than threshold degrees
def find_k_max_sources(bf_results, k):
    sorted_results = sorted(bf_results, key=lambda x: x[0], reverse=True)
    selected = []
    for power, th, ph in sorted_results:
        if all(np.sqrt((th - t) ** 2 + (ph - p) ** 2) >= 10.0 for _, t, p in selected):
            selected.append((power, th, ph))
            if len(selected) == k:
                break
    return selected

# Kalman filter class for 2D angle tracking
class KalmanFilter:
    def __init__(self, dt=0.032, process_var=1e-3, meas_var=1e-2):
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = process_var * np.eye(4)
        self.R = meas_var * np.eye(2)
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        z = z.reshape((2, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def current_state(self):
        return self.x[0, 0], self.x[1, 0]

# 3D plot
def plot_3d_scene(mic_positions, source_positions_cart, mic_radius, source_display_distance):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = mic_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = mic_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = mic_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.3)

    ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2], color='blue', label='Microphones', s=50)
    for i, pos in enumerate(source_positions_cart):
        ax.scatter(pos[0], pos[1], pos[2], color='red', s=100, marker='*', label='Source' if i == 0 else "")
        ax.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], color='red', linestyle='dashed')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Microphones (blue) and Detected Sources (red)')
    limit = source_display_distance * 1.2
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.legend()
    plt.show()

# Main
if __name__ == "__main__":
    n_mics = 16
    mic_radius = 0.15  # 15 cm sphere radius
    fs = 16000
    duration_s = 1.0   # 1 second
    c = 343.0
    k = 3              # number of sources to detect and track

    # Generate microphone positions
    mic_positions = generate_mic_positions_on_sphere(n_mics, mic_radius)

    # Define true source angles and random distances
    true_thetas = [40, 80, 120]
    true_phis  = [0, 90, 220]
    source_distance_factors = []
    signals = [np.zeros(int(duration_s * fs)) for _ in range(n_mics)]

    for t, p in zip(true_thetas, true_phis):
        dist_factor = random.uniform(10.0, 30.0)
        source_distance_factors.append(dist_factor)
        sim_sigs, _ = simulate_impulse_signals(mic_positions, fs, duration_s, t, p,
                                               noise=0.01, source_distance_factor=dist_factor)
        for i in range(n_mics):
            signals[i] += sim_sigs[i]

    # Bandpass filtering
    apply_bpf = True
    if apply_bpf:
        for i in range(n_mics):
            signals[i] = bandpass_filter(signals[i], 900, 1100, fs)

    frame_size = 1024
    hop_size = 512
    hop_size = 1024 
    n_frames = (len(signals[0]) - frame_size) // hop_size + 1

    # Initialize Kalman filters for each source
    kalman_filters = [KalmanFilter(dt=hop_size / fs) for _ in range(k)]
    tracked_positions = [[] for _ in range(k)]

    grid_theta = np.linspace(0, 180, 15)
    grid_phi = np.linspace(0, 360, 30)

    print("Starting multi-frame SRP-PHAT processing and source tracking:")

    for frame_idx in range(n_frames):
        frame_signals = [sig[frame_idx * hop_size:frame_idx * hop_size + frame_size] for sig in signals]

        bf_results = srp_phat_full(frame_signals, mic_positions, fs, grid_theta, grid_phi, c)

        k_best = find_k_max_sources(bf_results, k)

        print(f"\nFrame {frame_idx + 1}/{n_frames}:")

        for i, (power, th, ph) in enumerate(k_best):
            kalman_filters[i].predict()
            kalman_filters[i].update(np.array([th, ph]))
            th_est, ph_est = kalman_filters[i].current_state()
            tracked_positions[i].append((th_est, ph_est))
            print(f" Source {i + 1}: Smoothed Direction θ={th_est:.1f}°, φ={ph_est:.1f}°")

    # Final estimates and plot
    detected_sources_cart = []
    max_dist_factor = max(source_distance_factors)

    for i in range(k):
        th, ph = kalman_filters[i].current_state()
        dist_factor = source_distance_factors[i] if i < len(source_distance_factors) else max_dist_factor
        pos_cart = sph_to_cart(dist_factor * mic_radius, th, ph)
        detected_sources_cart.append(pos_cart)

    print("\nFinal tracked source positions:")
    for i, pos in enumerate(detected_sources_cart):
        dist = np.linalg.norm(pos)
        print(f" Source {i + 1}: Cartesian = {pos}, Distance ≈ {dist:.3f} m")

    plot_3d_scene(mic_positions, detected_sources_cart, mic_radius, max_dist_factor * mic_radius)

