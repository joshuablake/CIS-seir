"""Thin MCMC posterior .npz files to every STRIDE-th iteration, verify, delete the full file.

Usage: python thin_npz.py <dir> <stride> <n_tasks> <task_id> <staging_dir>

For each posterior_*.npz or posterior_*.npy.gz in <dir> assigned to this task (sorted, round-robin):
  1. load full chain (npz key arr_0, or gzipped .npy), keep rows 0, STRIDE, 2*STRIDE, ...
  2. write <name minus .npz>_thin<STRIDE>.npz to <staging_dir> (off the full quota)
  3. reload the staged file and check rows are bit-identical to the full chain's rows
  4. only then delete the full file, and move the thinned file into <dir>
Saved keys: samples, iteration (original row index), thin, full_shape.
"""
import os, sys, glob, gzip, shutil
import numpy as np

d, stride, n_tasks, task, staging = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
files = sorted(f for f in glob.glob(os.path.join(d, "posterior_*")) if "_thin" not in f
               and (f.endswith(".npz") or f.endswith(".npy.gz")))


def load_full(f):
    if f.endswith(".npy.gz"):
        with gzip.open(f, "rb") as g:
            return np.load(g, allow_pickle=False)
    with np.load(f, allow_pickle=False) as z:
        assert list(z.keys()) == ["arr_0"], (f, list(z.keys()))
        return z["arr_0"]



mine = files[task::n_tasks]
print(f"task {task}: {len(mine)} files", flush=True)
for f in mine:
    full = load_full(f)
    idx = np.arange(0, full.shape[0], stride)
    base = os.path.basename(f)
    for ext in (".npy.gz", ".npy.npz", ".npz"):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    out_name = base + f"_thin{stride}.npz"
    staged = os.path.join(staging, out_name)
    np.savez(staged, samples=full[idx], iteration=idx, thin=stride, full_shape=np.array(full.shape))
    with np.load(staged) as z:
        ok = (np.array_equal(z["samples"], full[idx]) and np.array_equal(z["iteration"], idx)
              and tuple(z["full_shape"]) == full.shape and z["samples"].dtype == full.dtype)
    if not ok:
        print(f"VERIFY FAILED {f}; full file kept", flush=True)
        continue
    os.remove(f)
    shutil.move(staged, os.path.join(d, out_name))
    print(f"ok {os.path.basename(f)} {full.shape} -> {len(idx)} rows", flush=True)
print("done", flush=True)
