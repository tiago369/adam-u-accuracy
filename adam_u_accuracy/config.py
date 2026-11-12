"""Constants shared by the model, controller and experiments.

Sources: PND Botics wiki (actuator datasheet), the PND SDK example `open_arm.py` (gains), and the
`pnd_mujoco` example `adam_U_open_arm.py` (alternative gains). Values not published by PND (friction,
backlash, stiffness, noise) are declared scenario levels, not measurements.
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_XML = os.path.join(ROOT, 'third_party', 'pnd_mujoco', 'pnd_robots', 'adam_u', 'adam_u.xml')
RESULTS = os.path.join(ROOT, 'results')

ARM = ['shoulderPitch', 'shoulderRoll', 'shoulderYaw', 'elbow', 'wristYaw', 'wristPitch', 'wristRoll']
SIDES = ['Left', 'Right']
# the 19 actuated non-finger joints, in the SDK's order
J19 = ['waistRoll', 'waistPitch', 'waistYaw', 'neckYaw', 'neckPitch'] + \
      [f'{j}_{s}' for s in SIDES for j in ARM]
ARM_IDX = {'Left': np.arange(5, 12), 'Right': np.arange(12, 19)}
S_MIRROR = np.array([1, -1, -1, 1, -1, 1, -1])  # q_Right = S * q_Left gives the mirror pose

# PD gains (kp, kd) for the 19 joints
GAINS = {
    'sdk': (np.array([405, 405, 205, 9, 9] + [18, 9, 9, 9, 9, 9, 9] * 2, float),
            np.array([6.1, 6.1, 4.1, .9, .9] + [.9] * 14)),
    'mujoco_ex': (np.array([405, 405, 205, 9, 9] + [180, 180, 9, 9, 9, 9, 9] * 2, float),
                  np.array([6.75, 6.75, 3.42, .9, .9] + [1.8, 1.8, .9, .9, .9, .9, .9] * 2)),
}

PEAK_TORQUE = np.array([60, 60, 17.5, 17.5, 6.2, 6.4, 6.4])            # N.m, per arm joint (wiki)
REFLECTED_INERTIA = np.array([0.157, 0.157, 0.0424, 0.0424, 0.0167, 0.0168, 0.0168])  # rotor J x 51^2, kg.m^2

TCP_OFFSET = (0.0, 0.0, -0.12)    # TCP site in the wristRoll frame (palm centre), m

# timing of the ISO 9283 cycle
DT = 0.001                 # simulation and control period, s
T_MOVE = 2.0               # quintic move duration, s
DWELL = 4.0                # dwell at each pose, s
WINDOW = (3.5, 4.0)        # measurement window inside the dwell, s after arrival
SETTLE = 3.0               # initial settle at P1, s
SEQUENCE = ['P5', 'P4', 'P3', 'P2', 'P1']   # ISO 9283 cycle order
N_WARMUP = 3               # cycles discarded before measuring

# sensing and actuation noise, always on in the plant (the source of cycle-to-cycle variation)
ENC_LSB = 2 * np.pi / (4000 * 51)   # motor-side 4000 counts/turn behind a 51:1 reducer, rad
ENC_NOISE_LSB = 1.0                  # white encoder noise, sigma in counts
TORQUE_NOISE = 0.02                  # white motor torque noise, sigma in N.m (current-loop ripple)
LATENCY_STEPS = 1                    # feedback delay, control periods

# virtual instruments: per-frame sigma (m, rad) and frames averaged per pose
INSTRUMENTS = {
    'mocap': dict(sig_p=0.2e-3, sig_r=np.radians(0.05), frames=60),
    'laser_tracker': dict(sig_p=0.015e-3, sig_r=np.radians(0.005), frames=10),
}
