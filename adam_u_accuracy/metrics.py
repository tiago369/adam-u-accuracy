"""ISO 9283:1998 pose characteristics, computed per pose and per arm.

Position, for n attained points x_j at one pose with commanded position x_c:
    AP_p = || mean(x_j) - x_c ||                       (pose accuracy, distance to the barycentre)
    RP_l = l_bar + 3 S_l,  l_j = || x_j - mean(x) ||   (pose repeatability)
Orientation: the error of each attained frame is the rotation vector of R_c^T R_j, i.e. small rotations (a, b, c)
about the commanded tool axes x, y, z:
    AP_a = mean(a_j) (likewise b, c),   RP_a = 3 S_a.
Multi-directional pose accuracy variation (approach from several directions, same commanded pose):
    vAP_p = max over direction pairs of || mean_h(x) - mean_k(x) ||.
RP also gets a percentile-bootstrap 95 % CI over cycles, because 3 S is an estimate from a finite sample.
"""
import numpy as np


def rotvec(R):
    """Rotation vectors (..., 3) of rotation matrices (..., 3, 3)."""
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(cos)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.where(th > 1e-12, th / (2 * np.sin(np.maximum(th, 1e-12))), 0.5)
    return w * s[..., None]


def rp_l(x):
    """x: (n, 3) attained points -> RP_l."""
    l = np.linalg.norm(x - x.mean(0), axis=-1)
    return l.mean() + 3 * l.std(ddof=1)


def bootstrap_ci(x, stat, n_boot=2000, rng=None, level=0.95):
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def pose_metrics(p, R, pc, Rc, ci=False, rng=None):
    """p: (C, P, A, 3) attained positions; R: (C, P, A, 3, 3); pc: (P, A, 3), Rc: (P, A, 3, 3) commanded.
    Returns arrays of shape (P, A) (mm, deg) plus aggregates."""
    C, P, A = p.shape[:3]
    bary = p.mean(0)
    dev = bary - pc
    ap = np.linalg.norm(dev, axis=-1) * 1e3
    rp = np.array([[rp_l(p[:, i, a]) for a in range(A)] for i in range(P)]) * 1e3
    e_rot = np.degrees(rotvec(np.einsum('paji,cpajk->cpaik', Rc, R)))      # (C, P, A, 3) about tool axes
    out = dict(AP_p=ap, AP_xyz=dev * 1e3, RP_l=rp,
               AP_abc=e_rot.mean(0), RP_abc=3 * e_rot.std(0, ddof=1),
               AP_mean=float(ap.mean()), AP_max=float(ap.max()), RP_mean=float(rp.mean()), RP_max=float(rp.max()))
    if ci:
        lo, hi = np.zeros((P, A)), np.zeros((P, A))
        for i in range(P):
            for a in range(A):
                lo[i, a], hi[i, a] = bootstrap_ci(p[:, i, a], rp_l, rng=rng)
        out['RP_l_ci'] = np.stack([lo, hi], -1) * 1e3
    return out


def vap(groups):
    """groups: list of (n_k, 3) attained points, one per approach direction -> vAP_p in mm."""
    bary = [g.mean(0) for g in groups]
    return 1e3 * max(np.linalg.norm(bi - bj) for i, bi in enumerate(bary) for bj in bary[i + 1:])


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return np.round(obj, 6).tolist()
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    return obj
