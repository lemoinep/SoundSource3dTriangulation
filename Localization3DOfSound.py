# Author(s): Dr. Patrick Lemoine

import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.signal import butter, lfilter
from pydub import AudioSegment
import os

SPEED_OF_SOUND = 343.0  # speed of sound (m/s)


def generate_sine_beep(duration_ms=500, freq_hz=1000, sample_rate=44100, volume_db=-10):
    """
    Generates a pure sine beep (mono) as an AudioSegment.
    """
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    samples = np.int16(sine_wave * 32767)
    audio_seg = AudioSegment(
        samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    return audio_seg + volume_db

def save_delayed_copies(audio_segment, delays_ms, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, delay in enumerate(delays_ms):
        silence = AudioSegment.silent(duration=delay)
        delayed = silence + audio_segment
        delayed = delayed[:len(audio_segment)]  # same duration as original
        path = os.path.join(output_dir, f'mic_{i + 1}.wav')
        delayed.export(path, format='wav')
        print(f"Generated file: {path} with delay {delay}ms")

def load_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        sr, data = wavfile.read(path)
    elif ext == '.mp3':
        audio = AudioSegment.from_mp3(path)
        sr = audio.frame_rate
        data = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            data = data.reshape((-1, 2))[:, 0]  # left channel
    else:
        raise ValueError("Unsupported audio format (only wav and mp3 supported)")
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648
    return sr, data

def bandpass_filter(data, lowcut, highcut, samplerate, order=4):
    """
    Simple Butterworth bandpass filter.
    """
    nyq = 0.5 * samplerate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def calculate_tdoa(sig_ref, sig_other, samplerate):
    corr = signal.correlate(sig_other, sig_ref, mode='full')
    lags = signal.correlation_lags(len(sig_other), len(sig_ref), mode='full')
    lag = lags[np.argmax(corr)]
    return lag / samplerate

def estimate_source_position(mic_positions, tdoas, sound_speed=SPEED_OF_SOUND):
    """
    Estimates 3D position from microphone positions and TDOAs (relative to mic 0).
    """
    n = len(mic_positions)
    if len(tdoas) != n - 1:
        raise ValueError("Number of TDOAs must be equal to number of microphones minus 1")
    p0 = mic_positions[0]
    A = []
    b = []
    for i in range(1, n):
        pi = mic_positions[i]
        di = tdoas[i - 1] * sound_speed
        A.append(2 * (pi - p0))
        b.append(di ** 2 - np.dot(pi, pi) + np.dot(p0, p0))
    A = np.array(A)
    b = np.array(b)
    pos, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return pos

def theta_phi_source_to_plane(mic_positions, source_pos):
    """
    Returns θ (elevation) and φ (azimuth) in degrees:
    - θ: angle between direction from plane to source and plane normal
    - φ: angle in plane from the vector (mic1->mic2)
    """
    u = mic_positions[1] - mic_positions[0]  # in-plane reference (X')
    v = mic_positions[2] - mic_positions[0]
    n = np.cross(u, v)
    n_hat = n / np.linalg.norm(n)
    center = np.mean(mic_positions, axis=0)
    s = source_pos - center
    if np.linalg.norm(s) == 0:
        return 0.0, 0.0
    s_hat = s / np.linalg.norm(s)
    # THETA: angle with normal
    dot_product = np.dot(n_hat, s_hat)
    theta_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    theta_deg = np.degrees(theta_rad)
    # Azimut : projection of s onto plane
    s_proj = s - np.dot(s, n_hat) * n_hat
    if np.linalg.norm(s_proj) == 0:
        phi_deg = 0.0  # source vertically aligned above/below the plane
    else:
        s_proj_hat = s_proj / np.linalg.norm(s_proj)
        u_hat = u / np.linalg.norm(u)  # reference
        dot_proj = np.dot(s_proj_hat, u_hat)
        phi_rad = np.arccos(np.clip(dot_proj, -1.0, 1.0))
        # Detect sense (CW or CCW) using plane normal
        sign = np.sign(np.dot(np.cross(u_hat, s_proj_hat), n_hat))
        if sign < 0:
            phi_rad = 2 * np.pi - phi_rad
        phi_deg = np.degrees(phi_rad) % 360
    return theta_deg, phi_deg

# --- Main program ---

def main():
    print("3D triangulation with 3 microphones, one placed higher on Z axis.")
    print("Options:")
    print("1 = Generate synthetic sounds with delays")
    print("2 = Load existing sounds")
    choice = input("Your choice (1 or 2)? ").strip()

    print("Do you want to apply noise filtering (bandpass filter)? [y/n]")
    filter_choice = input().strip().lower()
    apply_filter = filter_choice == 'y'

    mic_pos1 = np.array([-0.08, 0.0, 0.0]) # Mic 1 (left)
    mic_pos2 = np.array([0.08, 0.0, 0.0])  # Mic 2 (right)
    mic_pos3 = np.array([0.0, 0.0, 0.1])   # Mic 3 (top)
    mic_positions = np.vstack([mic_pos1, mic_pos2, mic_pos3])
    print(f"Microphone positions (m):\nMic 1: {mic_pos1}\nMic 2: {mic_pos2}\nMic 3: {mic_pos3}")

    if choice == "1":
        print("Generating synthetic sounds...")
        beep = generate_sine_beep()
        delays_ms = [0, 1, 2]  # example delays
        out_dir = "simulated_mics"
        save_delayed_copies(beep, delays_ms, out_dir)
        files = [os.path.join(out_dir, f"mic_{i + 1}.wav") for i in range(3)]
    elif choice == "2":
        files = []
        for i in range(3):
            path = input(f"Path to audio file for microphone {i + 1} (.wav or .mp3): ").strip()
            if not os.path.isfile(path):
                print(f"File not found: {path}")
                return
            files.append(path)
    else:
        print("Invalid choice, exiting.")
        return

    print("\nLoading audio files...")
    samplers = []
    signals = []
    for f in files:
        sr, data = load_audio(f)
        samplers.append(sr)
        signals.append(data)
        print(f"{f}: {len(data)} samples at {sr} Hz")

    if len(set(samplers)) != 1:
        print("Error: all files must have the same sampling rate.")
        return
    samplerate = samplers[0]
    min_len = min(map(len, signals))
    signals = [s[:min_len] for s in signals]

    # --- Noise filtering option ---
    if apply_filter:
        print("Applying bandpass filter (e.g., 900 Hz - 1100 Hz)...")
        lowcut = 900
        highcut = 1100
        filtered_signals = []
        for idx, sig in enumerate(signals):
            filtered = bandpass_filter(sig, lowcut, highcut, samplerate)
            filtered_signals.append(filtered)
            print(f"Signal {idx + 1} filtered.")
        signals = filtered_signals

    print("Calculating time delays (TDOA)...")
    tdoa_1 = calculate_tdoa(signals[0], signals[1], samplerate)
    tdoa_2 = calculate_tdoa(signals[0], signals[2], samplerate)

    print(f"TDOA mic2 vs mic1: {tdoa_1:.7f} s")
    print(f"TDOA mic3 vs mic1: {tdoa_2:.7f} s")

    pos_source = estimate_source_position(mic_positions, [tdoa_1, tdoa_2])
    print("\nEstimated source position (x, y, z) in meters:")
    print(pos_source)

    mid_point = np.mean(mic_positions, axis=0)
    dist_mid = np.linalg.norm(pos_source - mid_point)
    print(f"Distance from source to midpoint of microphones: {dist_mid:.3f} m")

    # --- Angle calculation section ---
    theta_deg, phi_deg = theta_phi_source_to_plane(mic_positions, pos_source)
    print(f"Elevation angle θ (with plane normal): {theta_deg:.2f} degrees")
    print(f"Azimuth angle φ (in plane, from Mic1->Mic2): {phi_deg:.2f} degrees")

if __name__ == "__main__":
    main()
