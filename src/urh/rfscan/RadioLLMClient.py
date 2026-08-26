import base64
import json
import logging
import os
import time
import urllib.request

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RADIOLLM_URL = "http://127.0.0.1:8737"

# complex formats URH supports that we know how to convert: (dtype, scale)
_FORMATS = {
    ".complex16s": (np.int8, 127.0),
    ".complex16u": (np.uint8, 127.0),
    ".cu8": (np.uint8, 127.0),
    ".cs8": (np.int8, 127.0),
    ".complex32s": (np.int16, 32767.0),
    ".complex64s": (None, 1.0),
}


class RadioLLMError(Exception):
    pass


def load_iq(path):
    """Read an IQ capture file -> (complex64 ndarray, meta dict)."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _FORMATS:
        raise RadioLLMError("unsupported IQ format: {0}".format(ext))
    dtype, scale = _FORMATS[ext]
    raw = np.fromfile(path, dtype=dtype)
    if raw.dtype != np.complex64:
        if raw.size % 2:
            raw = raw[:-1]
        iq = (raw[0::2].astype(np.float32) / scale
              + 1j * raw[1::2].astype(np.float32) / scale).astype(np.complex64)
    else:
        iq = raw
    return iq, {"format": ext, "samples": int(iq.size)}


def save_iq(path, iq):
    """Save complex64 array as .complex16s (URH native signed 8-bit)."""
    iq = np.asarray(iq, dtype=np.complex64)
    peak = max(abs(iq.real.max()), abs(iq.imag.max()), 1e-9)
    if peak > 1.0:
        iq = iq / peak
    inter = np.empty(2 * iq.size, dtype=np.int8)
    inter[0::2] = np.clip(iq.real * 127.0, -127, 127).astype(np.int8)
    inter[1::2] = np.clip(iq.imag * 127.0, -127, 127).astype(np.int8)
    inter.tofile(path)
    return path


def denoise(path, url=DEFAULT_RADIOLLM_URL, timeout=600,
            sample_rate=0.0, center_freq=0.0, progress=None):
    """Denoise an IQ capture file via the RadioLLM sidecar.

    Returns (out_path, stats_dict). Raises RadioLLMError on failure.
    """
    t0 = time.time()
    if not os.path.isfile(path):
        raise RadioLLMError("file not found: {0}".format(path))
    iq, meta = load_iq(path)
    logger.debug("RadioLLM: loaded {0} ({1} samples)".format(path, iq.size))

    body = json.dumps({
        "iq_b64": base64.b64encode(iq.tobytes()).decode(),
        "sample_rate": float(sample_rate),
        "center_freq": float(center_freq),
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/denoise", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            reply = json.loads(resp.read())
    except Exception as e:
        raise RadioLLMError("sidecar at {0} failed: {1}".format(url, e))

    den = np.frombuffer(base64.b64decode(reply["iq_b64"]), dtype=np.complex64)
    root, _ = os.path.splitext(path)
    out_path = root + "_denoised.complex16s"    
    save_iq(out_path, den)

    stats = {
        "samples": len(den),
        "elapsed_s": reply.get("elapsed_s"),
        "snr_proxy_db": reply.get("snr_proxy_db"),
        "out_path": out_path,
        "total_s": round(time.time() - t0, 2),
    }
    logger.info("RadioLLM: wrote {0} ({1})".format(out_path, stats))
    if progress is not None:
        progress(stats)
    return out_path, stats


def health(url=DEFAULT_RADIOLLM_URL, timeout=5):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health",
                                    timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}
