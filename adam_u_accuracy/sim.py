"""Closed-loop simulation of the ISO 9283 cycle on the error-injected plant.

Controller (runs at 1 kHz, one control period per simulation step), per actuated joint:
    tau = g_nominal(q_d) + kp (q_d - q_enc) + kd (dq_d - dq_enc) + torque noise
  * q_enc: the plant joint angle (motor side, i.e. before backlash and series compliance), delayed by LATENCY_STEPS,
    plus white noise, quantised to the encoder resolution; dq_enc: its filtered backward difference;
  * g_nominal: gravity torque computed on the NOMINAL model, never on the plant, so mass/CoM/payload errors appear
    as a static servo error delta_tau / kp, as they would on hardware;
  * q_d, dq_d: quintic interpolation between the frozen IK solutions, then constant during the dwell.
The finger position actuators hold the hands open (ctrl = 0).

At each pose, the measurement is the mean over the window [3.5, 4.0] s after arrival of:
  * the true TCP position and orientation of both hands (simulator ground truth, world frame = base frame);
  * the encoder readings of the 19 joints (used for encoder-based FK by the instruments module).
"""
import numpy as np
import mujoco

from .config import (J19, GAINS, DT, T_MOVE, DWELL, WINDOW, SETTLE, SEQUENCE, N_WARMUP,
                     ENC_LSB, ENC_NOISE_LSB, TORQUE_NOISE, LATENCY_STEPS)
from .model import Scenario, nominal_model, plant_model
from .poses import load_poses

VEL_CUTOFF_HZ = 50.0


class Nominal:
    """Nominal model: forward kinematics of the TCPs and gravity feed-forward."""

    def __init__(self):
        self.m = nominal_model()
        self.d = mujoco.MjData(self.m)
        self.qa = np.array([self.m.joint(n).qposadr[0] for n in J19])
        self.va = np.array([self.m.joint(n).dofadr[0] for n in J19])
        self.sites = [self.m.site('tcp_l').id, self.m.site('tcp_r').id]
        self._res = np.zeros(self.m.nv)

    def fk(self, q19):
        """TCP positions (2, 3) and rotation matrices (2, 3, 3) of both hands, world frame."""
        self.d.qpos[:] = 0
        self.d.qpos[self.qa] = q19
        mujoco.mj_kinematics(self.m, self.d)
        return (np.array([self.d.site_xpos[s] for s in self.sites]),
                np.array([self.d.site_xmat[s].reshape(3, 3) for s in self.sites]))

    def gravity(self, q19):
        self.d.qpos[:] = 0
        self.d.qpos[self.qa] = q19
        self.d.qvel[:] = 0
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_rne(self.m, self.d, 0, self._res)
        return self._res[self.va].copy()


def quintic(q0, q1, T, t):
    s = min(max(t / T, 0.0), 1.0)
    b = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    db = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
    return q0 + (q1 - q0) * b, (q1 - q0) * db


class Rig:
    """One virtual robot (plant) under one scenario, with its nominal controller."""

    def __init__(self, sc: Scenario, unit_seed: int = 0, nominal: Nominal | None = None, substeps: int = 1,
                 poses='palm_down'):
        """substeps > 1 refines the physics step (DT / substeps) while the controller stays at 1 kHz.
        poses: 'palm_down' | 'palm_in' | 'extra' | dict name -> 19-vector."""
        self.sc = sc
        self.substeps = substeps
        self.nom = nominal if nominal is not None else Nominal()
        self.m = plant_model(sc, np.random.default_rng(unit_seed))
        self.m.opt.timestep = DT / substeps
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qa = np.array([m.joint(n).qposadr[0] for n in J19])
        self.va = np.array([m.joint(n).dofadr[0] for n in J19])
        self.act = np.array([m.actuator(n).id for n in J19])
        self.sites = [m.site('tcp_l').id, m.site('tcp_r').id]
        self.kp, self.kd = GAINS[sc.gains]
        self.poses = load_poses(poses) if isinstance(poses, str) else {k: np.asarray(v) for k, v in poses.items()}
        # true output-side joint angle = motor angle - zero offset + passive joints (series spring, backlash)
        self.qpos0 = m.qpos0[self.qa].copy()
        self.extra = [(i, m.joint(f'{n}{suf}').qposadr[0]) for i, n in enumerate(J19) for suf in ('_k', '_bl')
                      if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'{n}{suf}') >= 0]
        self.wrench = None            # (site index 0/1, force (3,) N) applied at the TCP, or None
        self.alpha = DT / (DT + 1 / (2 * np.pi * VEL_CUTOFF_HZ))
        self.nsteps = 0

    # ------------------------------------------------------------------ low level
    def reset(self, q19):
        """Start at rest at q19: commanded angles are written as the motor-side (encoder) angles."""
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qa] = q19
        mujoco.mj_forward(self.m, self.d)
        self.hist = [q19.copy() for _ in range(LATENCY_STEPS)]   # delay line: the value read is LATENCY_STEPS periods old
        self.q_enc_prev = q19.copy()
        self.dq_f = np.zeros(19)

    def link_angles(self):
        q = self.d.qpos[self.qa] - self.qpos0
        for i, adr in self.extra:
            q[i] += self.d.qpos[adr]
        return q

    def _apply_wrench(self):
        d = self.d
        d.qfrc_applied[:] = 0
        if self.wrench is not None:
            s, F = self.wrench
            sid = self.sites[s]
            mujoco.mj_applyFT(self.m, d, np.asarray(F, float), np.zeros(3), d.site_xpos[sid].copy(),
                              self.m.site_bodyid[sid], d.qfrc_applied)

    def _encoder(self, rng):
        self.hist.append(self.d.qpos[self.qa].copy())
        q = self.hist.pop(0)
        if not self.sc.noise:
            return self.d.qpos[self.qa].copy()      # ideal, undelayed sensing
        q = q + ENC_NOISE_LSB * ENC_LSB * rng.standard_normal(19)
        return np.round(q / ENC_LSB) * ENC_LSB

    def segment(self, q0, q1, rng, T=T_MOVE, dwell=DWELL, record=True, window=WINDOW, trace=False):
        """Move q0 -> q1 in T s, dwell, and return the window-averaged measurement (or None).
        trace=True also returns the TCP positions of every control period of the move and dwell ('trace', (n, 2, 3))."""
        n_move, n_dwell = int(round(T / DT)), int(round(dwell / DT))
        lo, hi = int(round(window[0] / DT)), int(round(window[1] / DT))
        tr = np.zeros((n_move + n_dwell, 2, 3)) if trace else None
        g_hold = self.nom.gravity(q1)
        zero = np.zeros(19)
        tau_noise = (TORQUE_NOISE if self.sc.noise else 0.0) * rng.standard_normal((n_move + n_dwell, 19))
        acc_p, acc_R, acc_q, acc_l, n = np.zeros((2, 3)), np.zeros((2, 3, 3)), np.zeros(19), np.zeros(19), 0
        d = self.d
        for k in range(n_move + n_dwell):
            if k < n_move:
                qd, dqd = quintic(q0, q1, T, k * DT)
                g = self.nom.gravity(qd)
            else:
                qd, dqd, g = q1, zero, g_hold
            q_enc = self._encoder(rng)
            self.dq_f += self.alpha * ((q_enc - self.q_enc_prev) / DT - self.dq_f)
            self.q_enc_prev = q_enc
            d.ctrl[self.act] = g + self.kp * (qd - q_enc) + self.kd * (dqd - self.dq_f) + tau_noise[k]
            if self.wrench is not None:
                self._apply_wrench()
            mujoco.mj_step(self.m, d, self.substeps)
            self.nsteps += 1
            if trace:
                tr[k] = d.site_xpos[self.sites]
            j = k - n_move + 1                      # control periods since arrival
            if record and lo < j <= hi:
                acc_p += d.site_xpos[self.sites]
                acc_R += d.site_xmat[self.sites].reshape(2, 3, 3)
                acc_q += q_enc
                acc_l += self.link_angles()
                n += 1
        if not record:
            return dict(trace=tr) if trace else None
        R = acc_R / n
        R = np.array([u @ vt for u, _, vt in (np.linalg.svd(Ri) for Ri in R)])   # project the mean onto SO(3)
        out = dict(p=acc_p / n, R=R, q_enc=acc_q / n, q_link=acc_l / n)
        if trace:
            out['trace'] = tr
        return out

    # ------------------------------------------------------------------ ISO 9283 cycle
    def run_cycles(self, n_cycles, cycle_seed=0, n_warmup=N_WARMUP, sequence=SEQUENCE):
        """Settle at P1, run n_warmup + n_cycles cycles of `sequence`; returns arrays over the measured cycles:
        p (C, 5, 2, 3), R (C, 5, 2, 3, 3), q_enc and q_link (C, 5, 19)."""
        rng = np.random.default_rng(cycle_seed)
        P = self.poses
        self.reset(P['P1'])
        self.segment(P['P1'], P['P1'], rng, T=0.0, dwell=SETTLE, record=False)
        prev = 'P1'
        out = dict(p=[], R=[], q_enc=[], q_link=[])
        for c in range(n_warmup + n_cycles):
            meas = []
            for name in sequence:
                meas.append(self.segment(P[prev], P[name], rng, record=c >= n_warmup))
                prev = name
            if c >= n_warmup:
                for key in out:
                    out[key].append([mm[key] for mm in meas])
        return {k: np.array(v) for k, v in out.items()}

    def commanded(self, sequence=SEQUENCE):
        """Commanded TCP poses x_c = FK_nominal(q_c): p (5, 2, 3), R (5, 2, 3, 3)."""
        fk = [self.nom.fk(self.poses[n]) for n in sequence]
        return np.array([f[0] for f in fk]), np.array([f[1] for f in fk])
