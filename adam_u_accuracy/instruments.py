"""Virtual measurement instruments applied post hoc to the simulated states.

Each instrument maps the stored run (true TCP pose and encoder readings per cycle and pose) to the attained pose it
would report:
  * ground_truth   - simulator truth (the oracle);
  * mocap          - optical motion capture on a rigid wrist cuff: truth + white noise, sigma 0.2 mm / 0.05 deg per
                     frame, 60 frames averaged per pose (noise floor ~0.03 mm);
  * laser_tracker  - truth + sigma 0.015 mm per point, 10 points averaged;
  * encoder_fk     - forward kinematics of the NOMINAL model at the encoder readings: what the robot believes.
Registration of the external instrument to the robot base is assumed perfect (a stated limitation).
"""
import numpy as np

from .config import INSTRUMENTS


def _small_rotations(rvec):
    """Rotation matrices for an array of rotation vectors (..., 3), exact Rodrigues formula."""
    th = np.linalg.norm(rvec, axis=-1, keepdims=True)[..., None]
    k = np.where(th[..., 0] > 0, rvec / np.maximum(th[..., 0], 1e-300), 0.0)
    K = np.zeros(rvec.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def ground_truth(run, nominal=None, rng=None):
    return run['p'], run['R']


def external(name):
    cfg = INSTRUMENTS[name]

    def measure(run, nominal=None, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        p, R = run['p'], run['R']
        sp, sr = cfg['sig_p'] / np.sqrt(cfg['frames']), cfg['sig_r'] / np.sqrt(cfg['frames'])
        p_hat = p + sp * rng.standard_normal(p.shape)
        R_hat = R @ _small_rotations(sr * rng.standard_normal(p.shape))
        return p_hat, R_hat
    measure.__name__ = name
    return measure


def encoder_fk(run, nominal, rng=None):
    q = run['q_enc']                                  # (C, P, 19)
    p, R = np.zeros(q.shape[:2] + (2, 3)), np.zeros(q.shape[:2] + (2, 3, 3))
    for c in range(q.shape[0]):
        for i in range(q.shape[1]):
            p[c, i], R[c, i] = nominal.fk(q[c, i])
    return p, R


INSTRUMENT_FUNCS = {'ground_truth': ground_truth, 'mocap': external('mocap'),
                    'laser_tracker': external('laser_tracker'), 'encoder_fk': encoder_fk}
