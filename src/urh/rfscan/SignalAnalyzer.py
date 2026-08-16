import math

import numpy as np


def analyze_signal(iq8, center_freq, sample_rate, fft_averages=1):
    """Compute spectral features of a captured IQ window (numpy only).

    :param iq8: complex IQ samples, or an (N, 2) int8 array of signed I/Q
    :param center_freq: tuned center frequency in Hz
    :param sample_rate: sample rate in Hz
    :param fft_averages: number of FFT windows to power-average. Averaging
        lowers the noise variance (up to ~10*log10(N) dB SNR gain for a tone),
        which lets weak, far-away signals clear the detection threshold.
    :return: dict with fft freqs/db, detected peaks, bandwidth, noise floor
    """
    result = {
        "n_fft": 0,
        "center_freq": center_freq,
        "sample_rate": sample_rate,
        "noise_floor_db": None,
        "peaks": [],
        "bandwidth_hz": 0.0,
        "signal_rssi_db": None,
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

    n_avg = max(1, min(int(fft_averages), n // nfft))
    window = np.hanning(nfft)

    def spectrum_of(seg):
        seg = (seg - np.mean(seg)) * window
        return np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2

    if n_avg <= 1:
        power = spectrum_of(samples[:nfft])
    else:
        powers = [
            spectrum_of(samples[i * nfft : (i + 1) * nfft])
            for i in range(n_avg)
        ]
        power = np.mean(powers, axis=0)
    spec = np.sqrt(np.maximum(power, 1e-24))
    mag_db = 20.0 * np.log10(spec + 1e-12)
    freqs_hz = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate))
    bin_width = sample_rate / nfft

    result["n_fft"] = nfft
    result["freqs_hz"] = freqs_hz
    result["mag_db"] = mag_db
    result["noise_floor_db"] = float(np.percentile(mag_db, 50))
    result["fft_averages"] = n_avg

    # Smooth spectrum to suppress single-bin noise spikes while keeping narrow
    # CW tones. A boxcar wider than a few bins (e.g. nfft//1024) smears a thin
    # peak below the detection threshold and hides far-away signals.
    k = max(3, nfft // 16384)
    smooth = np.convolve(mag_db, np.ones(k) / k, mode="same")

    floor = result["noise_floor_db"]
    # Peak must clear the noise by a solid margin. The median-based floor is
    # robust: the largest pure-noise FFT spike stays ~9 dB above the median,
    # so +12 dB rejects it while genuine signals pass by tens of dB.
    # With power averaging the noise spikes shrink by ~10*log10(n_avg) dB, so
    # the threshold can be lowered accordingly to catch weaker far signals.
    threshold = floor + max(7.0, 12.0 - 10.0 * math.log10(max(1, n_avg)))
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
        # A genuine tone has a Hann main lobe of >= 2 bins at -6 dB; a
        # single-bin FFT noise fluctuation is only 1 bin wide.
        _, lo6, hi6 = _lobe_width(
            mag_db, idx, db, drop_db=6.0, bin_width=bin_width, nfft=nfft
        )
        if hi6 <= lo6:
            continue
        width, lo, hi = _lobe_width(
            mag_db, idx, db, drop_db=20.0, bin_width=bin_width, nfft=nfft
        )
        # Narrowband (signal-only) power within the peak's lobe, excluding
        # broadband noise. Window-power correction keeps the same relative
        # scale as the wideband RSSI (mean ADC power).
        signal_power = float(
            np.sum(np.abs(spec[lo:hi + 1]) ** 2)
        ) / (nfft * float(np.sum(window ** 2)))
        peaks.append(
            {
                "freq_rel_hz": freq_rel,
                "freq_abs_hz": center_freq + freq_rel,
                "freq_mhz": (center_freq + freq_rel) / 1e6,
                "db": db,
                "db_above_floor": db - floor,
                "width_hz": width,
                "signal_rssi_db": 10.0 * np.log10(max(signal_power, 1e-12)),
            }
        )
        if len(peaks) >= 8:
            break

    if peaks:
        result["bandwidth_hz"] = max(p["width_hz"] for p in peaks)
        result["signal_rssi_db"] = peaks[0]["signal_rssi_db"]
    result["peaks"] = peaks
    result["n_peaks"] = len(peaks)
    return result


def _lobe_width(mag_db, idx, peak_db, drop_db, bin_width, nfft):
    """Width of a peak's lobe at `drop_db` below its maximum (in Hz).

    Returns (width_hz, lo_bin, hi_bin) so callers can bound the peak's bins.
    """
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
    return (hi - lo + 1) * bin_width, lo, hi


def peaks_summary(analysis) -> str:
    """Short human-readable summary for the samples table."""
    n = analysis.get("n_peaks", 0)
    if n == 0:
        return "no peaks"
    top = analysis["peaks"][0]
    bw = analysis.get("bandwidth_hz", 0.0) / 1e3
    return "{0} peaks {1:.3f} MHz {2:.0f} kHz".format(n, top["freq_mhz"], bw)
