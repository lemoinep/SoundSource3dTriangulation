# Author(s): Dr. Patrick Lemoine

import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import random

# --- Spherical to Cartesian conversion ---

def sph_to_cart(r, theta_deg, phi_deg):
    theta = np.radians(theta_deg)  # inclination from +Z axis
    phi = np.radians(phi_deg)      # azimuth from +X axis
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

# --- Generate microphone positions uniformly on a sphere ---

def generate_mic_positions_on_sphere(n_mics, radius):
    indices = np.arange(0, n_mics, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n_mics)
    theta = np.pi * (1 + 5**0.5) * indices
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])

# --- Bandpass filter ---

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

# --- Simulate impulse signals from a source at specified direction and distance ---

def simulate_impulse_signals(mic_positions, fs, duration_s,
                             src_theta, src_phi, c=343.0, noise=0.01,
                             source_distance_factor=1.5):
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
        sample_delay = int((delay - min_delay) * fs)
        sig = np.zeros(n_samples)
        if sample_delay < n_samples:
            sig[sample_delay] = 1.0
        sig += noise * np.random.randn(n_samples)
        signals.append(sig)
    return signals, src_pos

# --- Simple SRP-PHAT beamforming (sum of correlations) ---

def srp_phat_bf(signals, mic_positions, fs, grid_theta, grid_phi, c=343.0):
    n_mics = mic_positions.shape[0]
    radius = np.linalg.norm(mic_positions[0])
    sig_length = signals[0].shape[0]
    fft_signals = [np.fft.rfft(sig, n=sig_length) for sig in signals]
    freqs = np.fft.rfftfreq(sig_length, 1. / fs)

    results = []
    for th, ph in itertools.product(grid_theta, grid_phi):
        steer_vec = sph_to_cart(radius * 3, th, ph)
        delays = []
        for mic_pos in mic_positions:
            vec = steer_vec - mic_pos
            tau = np.linalg.norm(vec) / c
            delays.append(tau)
        tau0 = delays[0]
        relative_delays = [tau - tau0 for tau in delays]

        steer_sum = np.zeros_like(fft_signals[0])
        for i in range(n_mics):
            phase_shift = np.exp(-2j * np.pi * freqs * relative_delays[i])
            steer_sum += fft_signals[i] * phase_shift / (np.abs(fft_signals[i]) + 1e-10)
        power = np.abs(np.sum(steer_sum))
        results.append((power, th, ph))
    return results

# --- Find top k strongest source directions with minimal duplication ---

def find_k_max_sources(bf_results, k):
    sorted_results = sorted(bf_results, key=lambda x: x[0], reverse=True)
    selected = []
    for power, th, ph in sorted_results:
        too_close = False
        for _, t_sel, p_sel in selected:
            distance = np.sqrt((th - t_sel)**2 + (ph - p_sel)**2)
            if distance < 10.0:  # degrees threshold to exclude close duplicates
                too_close = True
                break
        if not too_close:
            selected.append((power, th, ph))
            if len(selected) == k:
                break
    return selected

# --- 3D plot of microphones and detected sources ---

def plot_3d_scene(mic_positions, source_positions_cart, mic_radius, source_display_distance):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw microphone sphere wireframe
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = mic_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = mic_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = mic_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.3)

    # Plot microphones
    ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2],
               color='blue', label='Microphones', s=50)

    # Plot detected sources
    for i, pos in enumerate(source_positions_cart):
        ax.scatter(pos[0], pos[1], pos[2], color='red', s=100, marker='*',
                   label='Sources' if i == 0 else "")
        ax.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], color='red', linestyle='dashed')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Microphones (blue) and Detected Sources (red)')
    axis_limit = source_display_distance * 1.2
    ax.set_xlim([-axis_limit, axis_limit])
    ax.set_ylim([-axis_limit, axis_limit])
    ax.set_zlim([-axis_limit, axis_limit])
    ax.legend()
    plt.show()

# --- Main program ---

if __name__ == "__main__":
    n_mics = 16
    mic_radius = 0.15          # radius of microphone sphere (meters)
    fs = 16000                 # sampling rate (Hz)
    duration_s = 0.1           # signal duration (seconds)
    c = 343.0                  # speed of sound (m/s)
    k = 5                      # number of strongest sources to detect

    # Lists to hold random source distance factors for each source
    source_distance_factors = []

    # Generate mic positions
    mic_positions = generate_mic_positions_on_sphere(n_mics, mic_radius)

    # Define true source directions
    true_thetas = [40, 80, 120]    # inclination in degrees
    true_phis = [0, 90, 220]       # azimuth in degrees

    signals = [np.zeros(int(duration_s * fs)) for _ in range(n_mics)]

    # Simulate impulse signals for each source with random distance factor
    for t, p in zip(true_thetas, true_phis):
        dist_factor = random.uniform(10.0, 30.0)  # random distance factor
        source_distance_factors.append(dist_factor)
        sim_signals, _ = simulate_impulse_signals(
            mic_positions, fs, duration_s, t, p,
            noise=0.01, source_distance_factor=dist_factor)
        for i in range(n_mics):
            signals[i] += sim_signals[i]

    # Optional bandpass filtering
    apply_bandpass = True
    if apply_bandpass:
        for i in range(n_mics):
            signals[i] = bandpass_filter(signals[i], 900, 1100, fs)

    # Define grid for beamforming evaluation
    grid_theta = np.linspace(0, 180, 45)
    grid_phi = np.linspace(0, 360, 90)

    # Run beamforming
    bf_results = srp_phat_bf(signals, mic_positions, fs, grid_theta, grid_phi, c=c)

    # Find top k sources detected
    k_best_sources = find_k_max_sources(bf_results, k=k)

    # Print source positions using the corresponding stored distances
    print("\nDetected sources (Cartesian coordinates with realistic distances):")
    detected_sources_cart = []
    for i, (power, theta, phi) in enumerate(k_best_sources):
        # Use stored random source distance factor (or fallback to mean if out of range)
        dist_factor = source_distance_factors[i] if i < len(source_distance_factors) else np.mean(source_distance_factors)
        pos_cart = sph_to_cart(dist_factor * mic_radius, theta, phi)
        dist = np.linalg.norm(pos_cart)
        detected_sources_cart.append(pos_cart)
        print(f"Source {i+1}: θ={theta:.1f}° φ={phi:.1f}° | xyz = {pos_cart} | distance = {dist:.3f} m (simulated distance factor = {dist_factor:.2f})")

    # Plot the 3D scene
    # For display radius, use max factor to ensure all sources fit in view
    max_dist_factor = max(source_distance_factors) if source_distance_factors else 20.0
    plot_3d_scene(mic_positions, detected_sources_cart, mic_radius, max_dist_factor * mic_radius)
