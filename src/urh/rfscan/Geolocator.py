import numpy as np

EARTH_RADIUS_M = 6371000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in meters."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(min(1.0, a))))


def to_local_meters(lat0: float, lon0: float, lats, lons):
    """Convert lat/lon arrays to meters in a local tangent plane around (lat0, lon0)."""
    lat0_r = np.radians(lat0)
    x = np.radians(np.asarray(lons, dtype=float) - lon0) * np.cos(lat0_r) * EARTH_RADIUS_M
    y = np.radians(np.asarray(lats, dtype=float) - lat0) * EARTH_RADIUS_M
    return x, y


def from_local_meters(lat0: float, lon0: float, x, y):
    lat = lat0 + np.degrees(y / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(x / (np.cos(np.radians(lat0)) * EARTH_RADIUS_M))
    return float(lat), float(lon)


def _rssi_to_power(rssi_db):
    return np.power(10.0, np.asarray(rssi_db, dtype=float) / 10.0)


def weighted_centroid(samples):
    """Weighted centroid of (lat, lon, rssi_db) samples.

    Weights are linear power above the noise floor, so strong samples dominate.
    """
    lats = np.array([s[0] for s in samples], dtype=float)
    lons = np.array([s[1] for s in samples], dtype=float)
    rssi = np.array([s[2] for s in samples], dtype=float)

    power = _rssi_to_power(rssi)
    noise_floor = power.min()
    weights = np.clip(power - noise_floor, 0.0, None)

    if weights.sum() <= 0:
        weights = np.ones_like(weights)

    lat = float(np.average(lats, weights=weights))
    lon = float(np.average(lons, weights=weights))
    return lat, lon


def trilaterate(samples, n_fixed=2.0, fit_path_loss_exp=False, max_iter=200):
    """Fit emitter position via free-space path loss RSSI model.

    Model:  RSSI_i = P0 - 10*n*log10(d_i),  d_i = distance to emitter.
    Unknowns: emitter (x, y) in meters + transmit power P0 (dB), and optionally n.

    The position is searched with coordinate descent from several seeds, with P0
    solved in closed form (weighted mean) for each candidate position. Returns
    (lat, lon, P0, n, rms_error) or None if there are too few samples.
    """
    if len(samples) < 3:
        return None

    lats = np.array([s[0] for s in samples], dtype=float)
    lons = np.array([s[1] for s in samples], dtype=float)
    rssi = np.array([s[2] for s in samples], dtype=float)

    c_lat, c_lon = weighted_centroid(samples)
    xs, ys = to_local_meters(c_lat, c_lon, lats, lons)
    xs = xs.astype(float)
    ys = ys.astype(float)

    w = _rssi_to_power(rssi)
    w = w / w.max()

    def fit_p0(x, y, n):
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        d = np.sqrt(np.maximum(d2, 1e-9))
        z = rssi + 10.0 * n * np.log10(d)
        p0 = float(np.average(z, weights=w))
        resid = z - p0
        return p0, d, float(np.sum(w * resid**2))

    def refine(x, y, n, step0=50.0):
        best_cost = fit_p0(x, y, n)[2]
        step = step0
        for _ in range(max_iter):
            moved = False
            for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                cost = fit_p0(x + dx, y + dy, n)[2]
                if cost < best_cost:
                    x += dx
                    y += dy
                    best_cost = cost
                    moved = True
                    break
            if not moved:
                step *= 0.5
                if step < 0.5:
                    break
        return x, y, best_cost

    n_values = [n_fixed] if not fit_path_loss_exp else [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    best = None
    # Seeds: centroid, every sample (strong ones first), and a coarse grid.
    seeds = [(0.0, 0.0)]
    order = np.argsort(-w)
    for i in order[:8]:
        seeds.append((float(xs[i]), float(ys[i])))
    for gx in (-300, 0, 300):
        for gy in (-300, 0, 300):
            seeds.append((float(gx), float(gy)))

    for n in n_values:
        for sx, sy in seeds:
            x, y, cost = refine(sx, sy, n, step0=100.0)
            p0, d, cost = fit_p0(x, y, n)
            if best is None or cost < best["cost"]:
                best = {"x": x, "y": y, "n": n, "p0": p0, "d": d, "cost": cost}

    if best is None:
        return None

    resid = (best["p0"] - 10.0 * best["n"] * np.log10(np.maximum(best["d"], 1e-9))) - rssi
    rms_error = float(np.sqrt(np.mean(resid**2)))
    lat, lon = from_local_meters(c_lat, c_lon, best["x"], best["y"])
    return lat, lon, best["p0"], best["n"], rms_error


def estimate_confidence(rms_error: float) -> str:
    if rms_error is None:
        return "low"
    if rms_error < 3.0:
        return "high"
    if rms_error < 8.0:
        return "medium"
    return "low"
