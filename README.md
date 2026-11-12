# ADAM-U accuracy and repeatability in MuJoCo

Measuring the end-effector pose accuracy and repeatability of the PND Botics ADAM-U in MuJoCo.

## Install

### 1. Clone with submodules

The ADAM-U model and PND's SDK are pulled in as git submodules under `third_party/`:

| Submodule | Source | Used for |
|---|---|---|
| `third_party/pnd_mujoco` | <https://github.com/pndbotics/pnd_mujoco> | ADAM-U MJCF model (`pnd_robots/adam_u/scene.xml`) |
| `third_party/pnd_sdk_python` | <https://github.com/pndbotics/pnd_sdk_python> | Optional: PND's official simulator (`simulate_python`) |

```bash
git clone --recursive https://github.com/tiago369/adam-u-accuracy.git
cd adam-u-accuracy
# already cloned without --recursive?
git submodule update --init --recursive
```

### 2a. Python virtual environment (recommended)

Requires Python 3.10. The experiments need only `mujoco==3.2.0` and `numpy`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional, only to run PND's official simulator in `third_party/pnd_mujoco/simulate_python`
(installs the SDK in editable mode plus `cyclonedds`, `opencv-python` and `pygame`):

```bash
pip install -r requirements-sdk.txt
```

### 2b. Docker (headless, reproducible)

Pins Python 3.10 and MuJoCo 3.2.0 on any OS. Use it for long headless runs. The interactive MuJoCo viewer is easier from the venv.

```bash
docker build -t adam-u-accuracy .
docker run --rm adam-u-accuracy                     # smoke test
docker run --rm -v "$PWD/results:/app/results" adam-u-accuracy python <script>.py
```

### 3. Check the install

```bash
python -c "import mujoco; m = mujoco.MjModel.from_xml_path('third_party/pnd_mujoco/pnd_robots/adam_u/scene.xml'); print('mujoco', mujoco.__version__, 'nq', m.nq, 'nu', m.nu)"
# expected: mujoco 3.2.0 nq 50 nu 43
```

> Note: installing `requirements-sdk.txt` rewrites egg-info files tracked by the SDK repo. The submodule is set to `ignore = dirty`, so this does not show up as a change in this repository.
