# Headless, reproducible runtime for the experiment campaign.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl

# libgl/egl only matter if a script renders images; simulation itself needs none.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libegl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "-c", "import mujoco; m = mujoco.MjModel.from_xml_path('third_party/pnd_mujoco/pnd_robots/adam_u/scene.xml'); print('mujoco', mujoco.__version__, 'nq', m.nq, 'nu', m.nu)"]
