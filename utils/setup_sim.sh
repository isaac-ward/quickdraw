#!/usr/bin/env bash
# ==============================================================================================
# Link the robosuite/robocasa SIMULATOR into the container's venv. Run ONCE after any image rebuild.
#
#   docker compose exec app bash -lc "/app/src/../utils/setup_sim.sh"   # or from the host:
#   ./utils/setup_sim.sh --in-container
#
# WHY THIS IS NOT IN THE DOCKERFILE, unlike every other dependency (those live in pyproject.toml).
# robosuite and robocasa are EDITABLE installs of source checkouts under /caches/sim, which is a
# named docker VOLUME:
#   * The checkouts carry ~23 GB of robocasa scene assets. Baking that into the image would make every
#     rebuild pull 23 GB and every image layer carry it.
#   * /caches is not mounted at BUILD time, only at run time, so the Dockerfile physically cannot
#     reach them.
#   * robocasa needs a robosuite from SOURCE, not the PyPI wheel -- the wheel raises
#     `ManipulationEnv.__init__() got an unexpected keyword argument 'load_model_on_init'`.
# So the packages stay in the volume (which survives container recreation) and this script re-creates
# the venv's links to them, which do not. Idempotent: safe to re-run.
#
# What pyproject.toml DOES pin, because robocasa asserts on them at import and they are ordinary wheels:
#   numpy==2.2.5, mujoco==3.3.1  (exact allow-lists upstream -- not ranges), plus h5py, lxml,
#   opencv-python-HEADLESS (plain opencv-python needs GTK libs this image does not ship).
# ==============================================================================================
set -euo pipefail

if [ "${1:-}" != "--in-container" ] && [ ! -d /app/.venv ]; then
  exec docker compose exec -T app bash -lc "/app/utils/setup_sim.sh --in-container"
fi

SIM="${SIM_ROOT:-/caches/sim}"
VENV="${VIRTUAL_ENV:-/app/.venv}"
for d in "$SIM/robosuite" "$SIM/robocasa"; do
  [ -d "$d" ] || { echo "error: $d missing. The sim checkouts live in the /caches volume; if it was" >&2
                   echo "       recreated they must be re-cloned + assets re-downloaded (~23 GB)." >&2; exit 1; }
done

echo "linking robosuite + robocasa from $SIM into $VENV ..."
# --no-deps: their metadata would drag in mink (which pins numpy<2 and would break the numpy==2.2.5
# robocasa itself asserts) and a PyPI robosuite that shadows the source checkout. Everything they
# actually import is declared in pyproject.toml instead.
VIRTUAL_ENV="$VENV" uv pip install --quiet --no-deps -e "$SIM/robosuite" -e "$SIM/robocasa"

echo "verifying ..."
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md, os
import numpy, mujoco
print(f"   numpy {numpy.__version__}   mujoco {mujoco.__version__}")
assert numpy.__version__ == "2.2.5", "robocasa asserts numpy==2.2.5 exactly"
assert mujoco.__version__ == "3.3.1", "robocasa asserts mujoco==3.3.1 exactly"
os.environ.setdefault("MUJOCO_GL", "osmesa")
import robosuite, robocasa
from robosuite.environments.base import REGISTERED_ENVS
print(f"   robosuite {md.version('robosuite')} from {os.path.dirname(robosuite.__file__)}")
print(f"   robocasa  {md.version('robocasa')} -> {len(REGISTERED_ENVS)} registered envs")
assert "PrepareCoffee" in REGISTERED_ENVS, "robocasa kitchen envs did not register"
print("   OK")
PY
