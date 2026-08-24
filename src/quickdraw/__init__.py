"""quickdraw package.

Cap BLAS / OpenMP threads BEFORE numpy (hence OpenBLAS) loads — the thread-count env is read once, at library
load time, so it MUST be set before the first numpy import. `python -m quickdraw.<x>` runs this package
__init__ before any submodule imports numpy, so this is the reliable single chokepoint.

Why: OpenBLAS in this image is built for <=128 threads. On a many-core host (e.g. 80 cores) the sklearn / UMAP
reducers spawn nested BLAS x OpenMP threads that overflow OpenBLAS's memory-region limit and SEGFAULT — hit in
eval_interpret's t-SNE / UMAP on ~61k points. Capping avoids the oversubscription (GPU training is unaffected;
the heavy math there runs on cuBLAS, not OpenBLAS). setdefault so an explicit env still wins.
"""
import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "8")

# CUDA allocator, same reasoning: read ONCE when torch initializes its allocator, so it must be set before the
# first torch import. It lived in train_world_model.py, which meant anything that imported training.setup
# WITHOUT the entrypoint -- notably wizard/scripts/verify_autobatch.sh -- probed a DIFFERENT allocator than
# training uses. Measured cost of that mismatch: reserved/allocated fragmentation read 2-11% instead of the
# real 0.1-0.5%, peak-vs-batch looked SUPERLINEAR (fitted intercept -3.0GB vs the true +0.2GB) purely from
# block-rounding, and the sizer chose batch 15 at 81.4GB where the real allocator supports 17 at 87.4GB.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
