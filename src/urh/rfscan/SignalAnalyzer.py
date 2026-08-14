import numpy as np


def analyze_signal(iq8, center_freq, sample_rate):
    """Compute spectral features of a captured IQ window (numpy only).

    :param iq8: complex IQ samples, or an (N, 2) int8 array of signed I/Q
    :param center_freq: tuned center frequency in Hz
    :param sample_rate: sample rate in Hz
    :return: dict with fft freqs/db, detected peaks, bandwidth, noise floor
    """
    result = {
        "n_fft": 0,
        "center_freq": center_freq,
        "sample_rate": sample_rate,
        "noise_floor_db": None,
        "peaks": [],
        "bandwidth_hz": 0.0,
        "n_peaks": 0,
        "freqs_hz": None,
        "mag_db": None,
    }
    iq = np.asarray(iq8)
    if iq.ndim == 2:
        if iq.shape[0] < 128 or iq.shape[1] < 2:
            return result
        samples = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
    else:
        if iq.ndim != 1 or iq.shape[0] < 128:
            return result
        samples = iq.astype(np.complex64)
    n = len(samples)
    nfft = 1 << int(np.floor(np.log2(n)))
    nfft = min(nfft, 1 << 16)
    if nfft < 128:
        return result

    window = np.hanning(nfft)
    x = (samples[:nfft] - np.mean(samples[:nfft])) * window
    spec = np.fft.fftshift(np.fft.fft(x))
    mag_db = 20.0 * np.log10(np.abs(spec) + 1e-12)
    freqs_hz = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate))
    bin_width = sample_rate / nfft

    result["n_fft"] = nfft
    result["freqs_hz"] = freqs_hz
    result["mag_db"] = mag_db
    result["noise_floor_db"] = float(np.percentile(mag_db, 25))

    # Smooth spectrum to suppress single-bin noise spikes.
    k = max(3, nfft // 1024)
    smooth = np.convolve(mag_db, np.ones(k) / k, mode="same")

    floor = result["noise_floor_db"]
    threshold = floor + 12.0
    max_db = float(mag_db.max())
    prominence_limit = max_db - 30.0  # Hann first sidelobe is only ~-31.5 dB

    # Local maxima above threshold.
    candidates = []
    for i in range(1, nfft - 1):
        if (
            smooth[i] >= threshold
            and smooth[i] >= smooth[i - 1]
            and smooth[i] >= smooth[i + 1]
        ):
            candidates.append((float(mag_db[i]), float(freqs_hz[i]), i))

    # Drop window sidelobes / peaks that are weak relative to the strongest line.
    candidates = [c for c in candidates if c[0] >= prominence_limit]

    # Keep strongest, enforcing a minimum separation between peaks
    # (suppresses window sidelobes / adjacent lobes of one emitter).
    candidates.sort(reverse=True)
    min_sep = max(3, nfft // 512) * bin_width
    peaks = []
    for db, freq_rel, idx in candidates:
        if any(abs(freq_rel - p["freq_rel_hz"]) < min_sep for p in peaks):
            continue
        width = _lobe_width(mag_db, idx, db, drop_db=20.0, bin_width=bin_width, nfft=nfft)
        peaks.append(
            {
                "freq_rel_hz": freq_rel,
                "freq_abs_hz": center_freq + freq_rel,
                "freq_mhz": (center_freq + freq_rel) / 1e6,
                "db": db,
                "db_above_floor": db - floor,
                "width_hz": width,
            }
        )
        if len(peaks) >= 8:
            break

    if peaks:
        result["bandwidth_hz"] = max(p["width_hz"] for p in peaks)
    result["peaks"] = peaks
    result["n_peaks"] = len(peaks)
    return result


def _lobe_width(mag_db, idx, peak_db, drop_db, bin_width, nfft):
    """Width of a peak's lobe at `drop_db` below its maximum (in Hz)."""
    limit = peak_db - drop_db
    lo = idx
    hi = idx
    steps = 0
    max_steps = nfft // 8
    while lo > 0 and mag_db[lo - 1] >= limit and steps < max_steps:
        lo -= 1
        steps += 1
    steps = 0
    while hi < nfft - 1 and mag_db[hi + 1] >= limit and steps < max_steps:
        hi += 1
        steps += 1
    return (hi - lo + 1) * bin_width


def peaks_summary(analysis) -> str:
    """Short human-readable summary for the samples table."""
    n = analysis.get("n_peaks", 0)
    if n == 0:
        return "no peaks"
    top = analysis["peaks"][0]
    bw = analysis.get("bandwidth_hz", 0.0) / 1e3
    return "{0} peaks {1:.3f} MHz {2:.0f} kHz".format(n, top["freq_mhz"], bw)
