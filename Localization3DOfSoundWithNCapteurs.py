# Author(s): Dr. Patrick Lemoine

import numpy as np
from scipy.signal import butter, lfilter
import itertools

# --- Spherical geometry helpers ---

def sph_to_cart(r, theta_deg, phi_deg):
    theta = np.radians(theta_deg)  # inclination from +Z axis
    phi = np.radians(phi_deg)      # azimuth from +X axis
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def cart_to_sph(vec):
    x, y, z = vec
    r = np.linalg.norm(vec)
    theta = np.degrees(np.arccos(z / r)) if r > 0 else 0.
    phi = np.degrees(np.arctan2(y, x)) % 360
    return r, theta, phi

def generate_mic_positions_on_sphere(n, R):
    """Distributes n microphones evenly on a sphere of radius R using the Fibonacci method."""
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n)          # inclination
    theta = np.pi * (1 + 5**0.5) * indices    # azimuth
    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)
    return np.column_stack([x, y, z])         # shape (n, 3)

# --- Signal processing helpers ---

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    from scipy.signal import butter, lfilter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# --- [EXAMPLE] Simulate mono impulse from direction (theta, phi) ---

def simulate_impulse_signals(mic_positions, fs, duration_s,
                             src_theta, src_phi, c=343.0, noise=0.01):
    """
    For simplicity: simulate an impulse (dirac) source at (src_theta, src_phi) at 1.5 * R from the center.
    Returns [signals], time delays are computed based on position difference to each mic.
    """
    n_mics = mic_positions.shape[0]
    R = np.linalg.norm(mic_positions[0])
    src_dist = 1.5 * R  # source at 1.5x radius away from center
    src_pos = sph_to_cart(src_dist, src_theta, src_phi)  # (x,y,z)
    center = np.zeros(3)
    delays = []
    for mic in mic_positions:
        d = np.linalg.norm(src_pos - mic)
        delays.append(d / c)
    max_delay = max(delays)
    n_samples = int(duration_s * fs)
    signals = []
    for i, delay in enumerate(delays):
        t0 = int((delay - min(delays)) * fs)
        sig = np.zeros(n_samples)
        if t0 < n_samples:
            sig[t0] = 1.0
        sig += noise * np.random.randn(n_samples)
        signals.append(sig)
    return signals, src_pos

# --- SRP-PHAT BEAMFORMING SIMPLE (SUM OF CORRELATIONS) ---

def srp_phat_bf(signals, mic_positions, fs, grid_theta, grid_phi, c=343.0):
    """
    Simple Steered Response Power (SRP-PHAT)-like search over grid for one frame.
    Returns list (energy, theta, phi). Not proper PHAT weighting but simple peak search.
    """
    n_mics = mic_positions.shape[0]
    R = np.linalg.norm(mic_positions[0])
    signal_length = signals[0].shape[0]

    # Precompute FFTs
    fft_signals = [np.fft.rfft(sig, n=signal_length) for sig in signals]
    freqs = np.fft.rfftfreq(signal_length, 1. / fs)

    results = []
    for th, ph in itertools.product(grid_theta, grid_phi):
        steering_vec = sph_to_cart(R * 3, th, ph)  # Assume far away
        # delays = tau (seconds) for each mic (relative to the array center)
        delays = []
        for mic in mic_positions:
            vec = steering_vec - mic
            tau = np.linalg.norm(vec) / c
            delays.append(tau)
        # Reference: first mic
        tau0 = delays[0]
        relative_delays = [tau - tau0 for tau in delays]
        # Apply delays in frequency
        steer_sum = np.zeros_like(fft_signals[0])
        for i in range(n_mics):
            phase_shift = np.exp(-2j * np.pi * freqs * relative_delays[i])
            steer_sum += fft_signals[i] * phase_shift / np.abs(fft_signals[i] + 1e-10)
        power = np.abs(np.sum(steer_sum))  # sum up as power
        results.append((power, th, ph))
    return results

# --- Detection of strongest sources ---

def find_k_max_sources(bf_results, k):
    sorted_res = sorted(bf_results, key=lambda x: x[0], reverse=True)
    used = []
    out = []
    for power, th, ph in sorted_res:
        # simple maximal filter: avoid near-duplicate directions
        dist_ok = True
        for _, t2, p2 in out:
            d = np.sqrt((th - t2)**2 + (ph - p2)**2)
            if d < 10.0:  # degrees
                dist_ok = False
                break
        if dist_ok:
            out.append((power, th, ph))
            if len(out) == k:
                break
    return out

# --- PROGRAM ENTRY ---

if __name__ == "__main__":
    # PARAMETERS
    n_mics = 16
    R = 0.15        # sphere radius (meters)
    fs = 16000      # sample rate
    duration_s = 0.1
    c = 343.0       # speed of sound

    k = 5           # number of strongest sources to find

    # 1. Generate positions of n microphones on sphere
    mic_positions = generate_mic_positions_on_sphere(n_mics, R)

    # 2. [Example] Simulate signals for k sources (in real: load actual mic signals!)
    true_thetas = [40, 80, 120]
    true_phis = [0, 90, 220]
    signals = [np.zeros(int(duration_s * fs)) for _ in range(n_mics)]
    # Sum impulses from each source
    for t0, p0 in zip(true_thetas, true_phis):
        sim_signals, _ = simulate_impulse_signals(mic_positions, fs, duration_s, t0, p0, noise=0.01)
        for i in range(n_mics):
            signals[i] += sim_signals[i]

    # Optional bandpass
    apply_bpf = True
    if apply_bpf:
        for i in range(n_mics):
            signals[i] = bandpass_filter(signals[i], 900, 1100, fs)

    # 3. Set up grid for theta/phi scan (can use coarser grid for efficiency)
    grid_theta = np.linspace(0, 180, 45)
    grid_phi = np.linspace(0, 360, 90)

    # 4. Beamforming — find energetic peaks ("directions of arrival")
    bf_results = srp_phat_bf(signals, mic_positions, fs, grid_theta, grid_phi, c=c)

    # 5. Select k strongest sources (peak suppression to avoid duplicates)
    k_best = find_k_max_sources(bf_results, k=k)

    # 6. For each, report spherical and cartesian position (at R from center)
    print("\nDetected sources:")
for i, (power, th, ph) in enumerate(k_best):
    pos_cart = sph_to_cart(R, th, ph)
    dist = np.linalg.norm(pos_cart)
    print(f"Source {i+1}: θ={th:.1f}° φ={ph:.1f}° | xyz = {pos_cart} | distance = {dist:.3f} m")

    # Optionally: mapping to center of sphere (all positions given wrt center)
    # angle elevations are θ (inclination from Z+), φ azimuth CCW from X+

