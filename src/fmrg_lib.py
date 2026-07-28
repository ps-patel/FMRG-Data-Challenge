"""Pure-numpy toolkit for the NSF FMRG DED challenge (no scipy/sklearn available)."""
import io, zlib, struct
import numpy as np

# ---------------- MAT v5 reader (uncompressed + zlib-compressed elements) ----------------
_MI_DTYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32,
              6: np.uint32, 7: np.float32, 9: np.float64, 12: np.int64, 13: np.uint64}
_MX_NUMERIC = set(range(6, 16))  # mxDOUBLE_CLASS(6) .. mxUINT64_CLASS(15)


def _read_element(buf, pos):
    dtype_field, nbytes = struct.unpack_from('<II', buf, pos)
    if dtype_field >> 16:  # small data element
        nbytes = dtype_field >> 16
        dtype_id = dtype_field & 0xFFFF
        data = buf[pos + 4: pos + 4 + nbytes]
        return dtype_id, data, pos + 8
    dtype_id = dtype_field
    data = buf[pos + 8: pos + 8 + nbytes]
    advance = 8 + nbytes
    if advance % 8:
        advance += 8 - (advance % 8)
    return dtype_id, data, pos + advance


def _parse_matrix(data):
    pos = 0
    dt, flags, pos = _read_element(data, pos)
    array_class = flags[0]
    dt, dims_raw, pos = _read_element(data, pos)
    dims = np.frombuffer(dims_raw, np.int32)
    dt, name_raw, pos = _read_element(data, pos)
    name = name_raw.tobytes().decode('ascii', 'replace').strip('\x00') if isinstance(name_raw, np.ndarray) else name_raw.decode('ascii', 'replace').strip('\x00')
    if array_class not in _MX_NUMERIC:
        return name, None
    dt, real_raw, pos = _read_element(data, pos)
    npdt = _MI_DTYPES.get(dt)
    if npdt is None:
        return name, None
    arr = np.frombuffer(real_raw, npdt)
    if arr.size != int(np.prod(dims)):
        return name, None
    return name, arr.reshape(dims[::-1]).transpose()  # MATLAB column-major


def loadmat_v5(path):
    out = {}
    with open(path, 'rb') as f:
        buf = f.read()
    pos = 128
    while pos < len(buf) - 8:
        dtype_field, nbytes = struct.unpack_from('<II', buf, pos)
        payload = buf[pos + 8: pos + 8 + nbytes]
        if dtype_field == 15:  # miCOMPRESSED
            payload = zlib.decompress(payload)
            dtype_field2, nbytes2 = struct.unpack_from('<II', payload, 0)
            if dtype_field2 == 14:
                name, arr = _parse_matrix(payload[8:8 + nbytes2])
                if arr is not None:
                    out[name] = arr
        elif dtype_field == 14:  # miMATRIX
            name, arr = _parse_matrix(payload)
            if arr is not None:
                out[name] = arr
        adv = 8 + nbytes
        if adv % 8:
            adv += 8 - (adv % 8)
        pos += adv
    return out


# ---------------- Thermal helpers (mirrors starter code conventions) ----------------
COMMON_X_START_MM, COMMON_X_END_MM = 20.0, 100.0
MM_PER_FRAME = 10.0 / 50.0
N_FRAMES = int(round((COMMON_X_END_MM - COMMON_X_START_MM) / MM_PER_FRAME))  # 400


def find_thermal_array(mat):
    best = None
    for k, v in mat.items():
        a = np.squeeze(np.asarray(v))
        if a.ndim == 3 and np.issubdtype(a.dtype, np.number):
            score = a.size * (10 if 400 in a.shape else 1)
            if best is None or score > best[0]:
                best = (score, k, a)
    if best is None:
        raise ValueError('no 3D array')
    k, a = best[1], best[2]
    s = a.shape
    if s[0] == s[1] and s[2] != s[0]:
        a = np.moveaxis(a, 2, 0)
    elif s[1] != s[2]:
        a = np.moveaxis(a, int(np.argmax(s)), 0)
    return np.asarray(a, np.float32), k


def frame_score(frames, q=99.5):
    n = frames.shape[0]
    flat = frames.reshape(n, -1)
    return np.nanpercentile(flat, q, axis=1)


def detect_laser_interval(score):
    n = len(score)
    pre = score[:max(5, n // 10)]
    med = np.nanmedian(pre)
    mad = 1.4826 * np.nanmedian(np.abs(pre - med))
    thr = max(np.nanmin(score) + .2 * (np.nanmax(score) - np.nanmin(score)), med + 8 * max(mad, 1e-12))
    mask = score > thr
    idx = np.flatnonzero(mask)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks] + 1, idx[-1] + 1]
    j = int(np.argmax(stops - starts))
    return int(starts[j]), int(stops[j]), thr


# ---------------- Wyko ASC loader (pandas-based, fast) ----------------
def load_wyko_asc(path, crop=True):
    import pandas as pd
    header = {}
    with open(path, 'r', errors='replace') as f:
        for i, line in enumerate(f):
            p = line.strip().split()
            if len(p) >= 3 and p[0].lower() == 'x' and p[1].lower() == 'size':
                header['x_size'] = int(float(p[2]))
            elif len(p) >= 3 and p[0].lower() == 'y' and p[1].lower() == 'size':
                header['y_size'] = int(float(p[2]))
            elif p and p[0].lower() == 'pixel_size':
                header['pixel_size_mm'] = float(p[-1])
            if p and p[0].upper() == 'RAW_DATA':
                header['data_start_line'] = i + 1
                break
    xs, ys = header['x_size'], header['y_size']
    pix = header.get('pixel_size_mm', 0.003982)
    df = pd.read_csv(path, sep='\t', header=None, skiprows=header['data_start_line'],
                     usecols=[2], na_values=['Bad'], dtype=np.float64,
                     nrows=xs * ys, engine='c')
    z = df.values[:, 0].astype(np.float32) * 1e-6  # nm -> mm
    n = xs * ys
    if z.size < n:
        z = np.r_[z, np.full(n - z.size, np.nan, np.float32)]
    Z = z[:n].reshape(xs, ys).T  # (y, x_local)
    x_local = np.arange(xs) * pix
    y_mm = np.arange(ys) * pix
    x_actual = 100.0 - x_local
    order = np.argsort(x_actual)
    Z = Z[:, order]
    x_actual = x_actual[order]
    if crop:
        m = (x_actual >= COMMON_X_START_MM) & (x_actual <= COMMON_X_END_MM)
        Z, x_actual = Z[:, m], x_actual[m]
    return {'Z_mm': Z, 'x_mm': x_actual, 'y_mm': y_mm, 'header': header}


def robust_plane_detrend(Z, x, y, sx=40, sy=2, exclude_mask=None):
    Zs = Z[::sy, ::sx]
    Xs, Ys = np.meshgrid(x[::sx], y[::sy])
    z = Zs.ravel()
    A = np.c_[Xs.ravel(), Ys.ravel(), np.ones(Xs.size)]
    valid = np.isfinite(z)
    if exclude_mask is not None:
        valid &= ~exclude_mask[::sy, ::sx].ravel()
    keep = valid.copy()
    coef = None
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        r = z - A @ coef
        lo, hi = np.nanpercentile(r[valid], [5, 95])
        knew = valid & (r >= lo) & (r <= hi)
        if knew.sum() < 100:
            break
        keep = knew
    plane = coef[0] * x[None, :] + coef[1] * y[:, None] + coef[2]
    return Z - plane, coef
